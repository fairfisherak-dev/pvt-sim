"""Standalone true boiling point (TBP) assay kernel.

The standalone TBP surface accepts ordered cut-resolved assay data and returns
an auditable assay summary. The broadened runtime contract now admits:

- single-carbon cuts such as ``C7``
- interval cuts such as ``C7-C9``
- gapped but ordered/non-overlapping cut sequences
- optional cut boiling points when supplied explicitly
- optional boiling-point estimation from ``MW`` and ``SG`` when possible
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Any, Mapping, Sequence

import numpy as np
from numpy.typing import NDArray

from ..core.errors import ValidationError


@dataclass(frozen=True)
class TBPAssayCut:
    """Single TBP cut accepted by the standalone assay kernel."""

    name: str
    carbon_number: int
    carbon_number_end: int
    mole_fraction: float
    molecular_weight_g_per_mol: float
    specific_gravity: float | None = None
    boiling_point_k: float | None = None

    @property
    def carbon_range_label(self) -> str:
        if self.carbon_number_end == self.carbon_number:
            return f"C{self.carbon_number}"
        return f"C{self.carbon_number}-C{self.carbon_number_end}"


@dataclass(frozen=True)
class TBPCutResult:
    """Derived cut-level result on both mole and mass assay bases."""

    name: str
    carbon_number: int
    carbon_number_end: int
    mole_fraction: float
    normalized_mole_fraction: float
    cumulative_mole_fraction: float
    molecular_weight_g_per_mol: float
    normalized_mass_fraction: float
    cumulative_mass_fraction: float
    specific_gravity: float | None = None
    boiling_point_k: float | None = None
    boiling_point_source: str | None = None

    @property
    def carbon_range_label(self) -> str:
        if self.carbon_number_end == self.carbon_number:
            return f"C{self.carbon_number}"
        return f"C{self.carbon_number}-C{self.carbon_number_end}"


@dataclass(frozen=True)
class TBPResult:
    """Standalone TBP assay summary."""

    cut_start: int
    cut_end: int
    cuts: tuple[TBPCutResult, ...]
    z_plus: float
    mw_plus_g_per_mol: float
    carbon_numbers: NDArray[np.int64]
    normalized_mole_fractions: NDArray[np.float64]
    cumulative_mole_fractions: NDArray[np.float64]
    normalized_mass_fractions: NDArray[np.float64]
    cumulative_mass_fractions: NDArray[np.float64]
    boiling_points_k: NDArray[np.float64]

    @property
    def cut_names(self) -> tuple[str, ...]:
        return tuple(cut.name for cut in self.cuts)

    @property
    def cumulative_mole_percent(self) -> NDArray[np.float64]:
        return self.cumulative_mole_fractions * 100.0

    @property
    def cumulative_mass_percent(self) -> NDArray[np.float64]:
        return self.cumulative_mass_fractions * 100.0

    @property
    def has_boiling_point_curve(self) -> bool:
        return bool(np.isfinite(self.boiling_points_k).any())


def simulate_tbp(
    cuts: Sequence[object],
    *,
    cut_start: int | None = None,
) -> TBPResult:
    """Build a standalone TBP assay summary from cut-resolved input.

    Parameters
    ----------
    cuts : sequence
        Ordered TBP cuts. Each item may be a mapping with `name`, `z`, and
        `mw` keys or an object exposing equivalent attributes. Optional cut
        specific gravity may be provided as `sg` or `specific_gravity`. Optional
        boiling point may be provided as ``tb_k`` / ``boiling_point_k`` (or the
        equivalent ``*_c`` / ``*_f`` aliases).
    cut_start : int, optional
        Expected first carbon number in the sequence. When omitted, the first
        cut defines the assay start.

    Returns
    -------
    TBPResult
        Cut-resolved cumulative mole- and mass-yield curves plus derived
        `z_plus` and `mw_plus_g_per_mol`.

    Notes
    -----
    The current runtime still treats TBP as a standalone assay workflow. When
    boiling points are present, or can be estimated from ``MW`` and ``SG``,
    the result also preserves cut-level boiling-point points for reporting.
    """
    if isinstance(cuts, (str, bytes, bytearray)):
        raise ValidationError(
            "TBP cuts must be provided as a sequence of cut definitions.",
            parameter="cuts",
            value=type(cuts).__name__,
        )

    cut_items = tuple(cuts)
    if len(cut_items) == 0:
        raise ValidationError(
            "TBP cuts must be a non-empty list.",
            parameter="cuts",
        )

    assay_cuts = tuple(_coerce_cut(cut_obj, index=index) for index, cut_obj in enumerate(cut_items))
    expected_cut_start = assay_cuts[0].carbon_number if cut_start is None else _as_positive_int(cut_start, "cut_start")
    _validate_cut_sequence(assay_cuts, cut_start=expected_cut_start)

    z_plus = float(sum(cut.mole_fraction for cut in assay_cuts))
    mole_fractions = np.asarray([cut.mole_fraction for cut in assay_cuts], dtype=np.float64)
    normalized_mole_fractions = mole_fractions / z_plus
    cumulative_mole_fractions = np.cumsum(normalized_mole_fractions)

    raw_mass_basis = np.asarray(
        [cut.mole_fraction * cut.molecular_weight_g_per_mol for cut in assay_cuts],
        dtype=np.float64,
    )
    total_mass_basis = float(raw_mass_basis.sum())
    if not isfinite(total_mass_basis) or total_mass_basis <= 0.0:
        raise ValidationError(
            "TBP cuts must sum to a positive assay mass basis.",
            parameter="cuts",
            value=total_mass_basis,
        )
    normalized_mass_fractions = raw_mass_basis / total_mass_basis
    cumulative_mass_fractions = np.cumsum(normalized_mass_fractions)

    boiling_points_k = np.full(len(assay_cuts), np.nan, dtype=np.float64)
    cut_results = tuple(
        _build_cut_result(
            cut=cut,
            index=index,
            normalized_mole_fraction=float(normalized_mole_fractions[index]),
            cumulative_mole_fraction=float(cumulative_mole_fractions[index]),
            normalized_mass_fraction=float(normalized_mass_fractions[index]),
            cumulative_mass_fraction=float(cumulative_mass_fractions[index]),
            boiling_points_k=boiling_points_k,
        )
        for index, cut in enumerate(assay_cuts)
    )

    return TBPResult(
        cut_start=expected_cut_start,
        cut_end=assay_cuts[-1].carbon_number_end,
        cuts=cut_results,
        z_plus=z_plus,
        mw_plus_g_per_mol=float(total_mass_basis / z_plus),
        carbon_numbers=np.asarray([cut.carbon_number for cut in assay_cuts], dtype=np.int64),
        normalized_mole_fractions=normalized_mole_fractions,
        cumulative_mole_fractions=cumulative_mole_fractions,
        normalized_mass_fractions=normalized_mass_fractions,
        cumulative_mass_fractions=cumulative_mass_fractions,
        boiling_points_k=boiling_points_k,
    )


def _coerce_cut(cut_obj: object, *, index: int) -> TBPAssayCut:
    prefix = f"cuts[{index}]"

    if isinstance(cut_obj, Mapping):
        name = _as_non_empty_str(_get_required_mapping_value(cut_obj, "name", prefix), f"{prefix}.name")
        carbon_number_value = cut_obj.get("carbon_number")
        carbon_number_end_value = cut_obj.get("carbon_number_end")
        mole_fraction_value, mole_fraction_parameter = _get_required_alias(
            cut_obj,
            prefix,
            aliases=(("z", f"{prefix}.z"), ("mole_fraction", f"{prefix}.mole_fraction")),
        )
        molecular_weight_value, molecular_weight_parameter = _get_required_alias(
            cut_obj,
            prefix,
            aliases=(("mw", f"{prefix}.mw"), ("molecular_weight_g_per_mol", f"{prefix}.molecular_weight_g_per_mol")),
        )
        specific_gravity_value = _get_optional_alias(cut_obj, "sg", "specific_gravity")
        boiling_point_k = _get_optional_temperature_alias(cut_obj, prefix)
    else:
        name = _as_non_empty_str(_get_required_object_attr(cut_obj, "name", prefix), f"{prefix}.name")
        carbon_number_value = _get_optional_object_attr(cut_obj, "carbon_number")
        carbon_number_end_value = _get_optional_object_attr(cut_obj, "carbon_number_end")
        mole_fraction_value, mole_fraction_parameter = _get_required_object_alias(
            cut_obj,
            prefix,
            aliases=(("z", f"{prefix}.z"), ("mole_fraction", f"{prefix}.mole_fraction")),
        )
        molecular_weight_value, molecular_weight_parameter = _get_required_object_alias(
            cut_obj,
            prefix,
            aliases=(("mw", f"{prefix}.mw"), ("molecular_weight_g_per_mol", f"{prefix}.molecular_weight_g_per_mol")),
        )
        specific_gravity_value = _get_optional_object_attr(cut_obj, "sg", "specific_gravity")
        boiling_point_k = _get_optional_object_temperature_alias(cut_obj, prefix)

    parsed_carbon_number, parsed_carbon_number_end = _parse_tbp_cut_name(name, f"{prefix}.name")
    if carbon_number_value is None:
        carbon_number = parsed_carbon_number
    else:
        carbon_number = _as_positive_int(carbon_number_value, f"{prefix}.carbon_number")
        if carbon_number != parsed_carbon_number:
            raise ValidationError(
                "TBP cut carbon_number must match the numeric suffix in name.",
                parameter=f"{prefix}.carbon_number",
                value=carbon_number,
            )
    if carbon_number_end_value is None:
        carbon_number_end = parsed_carbon_number_end
    else:
        carbon_number_end = _as_positive_int(carbon_number_end_value, f"{prefix}.carbon_number_end")
        if carbon_number_end != parsed_carbon_number_end:
            raise ValidationError(
                "TBP cut carbon_number_end must match the numeric suffix/range in name.",
                parameter=f"{prefix}.carbon_number_end",
                value=carbon_number_end,
            )
    if carbon_number_end < carbon_number:
        raise ValidationError(
            "TBP cut carbon_number_end must be >= carbon_number.",
            parameter=f"{prefix}.carbon_number_end",
            value=carbon_number_end,
        )

    mole_fraction = _as_positive_float(
        mole_fraction_value,
        mole_fraction_parameter,
        "TBP cut mole fraction must be positive.",
    )
    molecular_weight = _as_positive_float(
        molecular_weight_value,
        molecular_weight_parameter,
        "TBP cut molecular weight must be positive.",
    )

    specific_gravity: float | None = None
    if specific_gravity_value is not None:
        specific_gravity = _as_positive_float(
            specific_gravity_value,
            f"{prefix}.specific_gravity",
            "TBP cut specific gravity must be positive when provided.",
        )
    if boiling_point_k is not None:
        boiling_point_k = _as_positive_float(
            boiling_point_k,
            f"{prefix}.boiling_point_k",
            "TBP cut boiling point must be positive when provided.",
        )

    return TBPAssayCut(
        name=name,
        carbon_number=carbon_number,
        carbon_number_end=carbon_number_end,
        mole_fraction=mole_fraction,
        molecular_weight_g_per_mol=molecular_weight,
        specific_gravity=specific_gravity,
        boiling_point_k=boiling_point_k,
    )


def _validate_cut_sequence(cuts: Sequence[TBPAssayCut], *, cut_start: int) -> None:
    seen_names: set[str] = set()
    previous_cut_end: int | None = None

    for index, cut in enumerate(cuts):
        parameter = f"cuts[{index}].name"
        if cut.name in seen_names:
            raise ValidationError(
                "TBP cut names must be unique.",
                parameter=parameter,
                value=cut.name,
            )
        if cut.carbon_number < cut_start:
            raise ValidationError(
                "TBP cuts must not start below cut_start.",
                parameter=parameter,
                value=cut.name,
                cut_start=cut_start,
            )
        if previous_cut_end is None:
            if cut.carbon_number != cut_start:
                raise ValidationError(
                    "The first TBP cut must start at cut_start.",
                    parameter=parameter,
                    value=cut.name,
                    cut_start=cut_start,
                )
        elif cut.carbon_number <= previous_cut_end:
            raise ValidationError(
                "TBP cuts must be ordered, non-overlapping, and strictly increasing.",
                parameter=parameter,
                value=cut.name,
            )
        seen_names.add(cut.name)
        previous_cut_end = cut.carbon_number_end


def _parse_tbp_cut_name(name: str, parameter: str) -> tuple[int, int]:
    normalized = name.strip().replace(" ", "").upper()
    single_match = _match_single_cut_name(normalized)
    if single_match is not None:
        carbon_number = int(single_match)
        return carbon_number, carbon_number

    range_match = _match_cut_range_name(normalized)
    if range_match is None:
        raise ValidationError(
            "TBP cut name must look like 'C7' or 'C7-C9'.",
            parameter=parameter,
            value=name,
        )
    carbon_start = int(range_match[0])
    carbon_end = int(range_match[1])
    if carbon_end < carbon_start:
        raise ValidationError(
            "TBP cut range end must be >= range start.",
            parameter=parameter,
            value=name,
        )
    return carbon_start, carbon_end


def _match_single_cut_name(normalized: str) -> str | None:
    if len(normalized) < 2 or not normalized.startswith("C") or not normalized[1:].isdigit():
        return None
    return normalized[1:]


def _match_cut_range_name(normalized: str) -> tuple[str, str] | None:
    if "-" not in normalized or not normalized.startswith("C"):
        return None
    left, right = normalized.split("-", maxsplit=1)
    if not left.startswith("C") or not left[1:].isdigit():
        return None
    if right.startswith("C"):
        right = right[1:]
    if not right.isdigit():
        return None
    return left[1:], right


def _build_cut_result(
    *,
    cut: TBPAssayCut,
    index: int,
    normalized_mole_fraction: float,
    cumulative_mole_fraction: float,
    normalized_mass_fraction: float,
    cumulative_mass_fraction: float,
    boiling_points_k: NDArray[np.float64],
) -> TBPCutResult:
    boiling_point_k, boiling_point_source = _resolve_cut_boiling_point(cut)
    if boiling_point_k is not None:
        boiling_points_k[index] = boiling_point_k
    return TBPCutResult(
        name=cut.name,
        carbon_number=cut.carbon_number,
        carbon_number_end=cut.carbon_number_end,
        mole_fraction=cut.mole_fraction,
        normalized_mole_fraction=normalized_mole_fraction,
        cumulative_mole_fraction=cumulative_mole_fraction,
        molecular_weight_g_per_mol=cut.molecular_weight_g_per_mol,
        normalized_mass_fraction=normalized_mass_fraction,
        cumulative_mass_fraction=cumulative_mass_fraction,
        specific_gravity=cut.specific_gravity,
        boiling_point_k=boiling_point_k,
        boiling_point_source=boiling_point_source,
    )


def _resolve_cut_boiling_point(cut: TBPAssayCut) -> tuple[float | None, str | None]:
    if cut.boiling_point_k is not None:
        return cut.boiling_point_k, "input"
    if cut.specific_gravity is None:
        return None, None

    from ..correlations.boiling_point import estimate_Tb

    return (
        float(
            estimate_Tb(
                MW=cut.molecular_weight_g_per_mol,
                SG=cut.specific_gravity,
            )
        ),
        "estimated_soreide",
    )


def _get_required_mapping_value(mapping: Mapping[str, Any], key: str, prefix: str) -> Any:
    if key not in mapping:
        raise ValidationError(
            f"Missing required TBP field '{prefix}.{key}'.",
            parameter=f"{prefix}.{key}",
        )
    return mapping[key]


def _get_required_alias(
    mapping: Mapping[str, Any],
    prefix: str,
    *,
    aliases: Sequence[tuple[str, str]],
) -> tuple[Any, str]:
    for key, parameter in aliases:
        if key in mapping:
            return mapping[key], parameter
    alias_names = ", ".join(key for key, _ in aliases)
    raise ValidationError(
        f"Missing required TBP field at '{prefix}' (expected one of: {alias_names}).",
        parameter=prefix,
    )


def _get_optional_alias(mapping: Mapping[str, Any], *keys: str) -> Any | None:
    for key in keys:
        if key in mapping:
            return mapping[key]
    return None


def _get_optional_temperature_alias(mapping: Mapping[str, Any], prefix: str) -> float | None:
    if "tb_k" in mapping:
        return _as_positive_float(mapping["tb_k"], f"{prefix}.tb_k", "TBP cut boiling point must be positive.")
    if "boiling_point_k" in mapping:
        return _as_positive_float(
            mapping["boiling_point_k"],
            f"{prefix}.boiling_point_k",
            "TBP cut boiling point must be positive.",
        )
    if "tb_c" in mapping:
        return _as_positive_float(mapping["tb_c"], f"{prefix}.tb_c", "TBP cut boiling point must be positive.") + 273.15
    if "boiling_point_c" in mapping:
        return (
            _as_positive_float(
                mapping["boiling_point_c"],
                f"{prefix}.boiling_point_c",
                "TBP cut boiling point must be positive.",
            )
            + 273.15
        )
    if "tb_f" in mapping:
        return (_as_positive_float(mapping["tb_f"], f"{prefix}.tb_f", "TBP cut boiling point must be positive.") - 32.0) * (5.0 / 9.0) + 273.15
    if "boiling_point_f" in mapping:
        return (
            (_as_positive_float(
                mapping["boiling_point_f"],
                f"{prefix}.boiling_point_f",
                "TBP cut boiling point must be positive.",
            ) - 32.0)
            * (5.0 / 9.0)
            + 273.15
        )
    return None


def _get_required_object_attr(obj: object, attr: str, prefix: str) -> Any:
    if not hasattr(obj, attr):
        raise ValidationError(
            f"TBP cut objects must expose '{attr}' at '{prefix}'.",
            parameter=f"{prefix}.{attr}",
            value=type(obj).__name__,
        )
    return getattr(obj, attr)


def _get_required_object_alias(
    obj: object,
    prefix: str,
    *,
    aliases: Sequence[tuple[str, str]],
) -> tuple[Any, str]:
    for attr, parameter in aliases:
        if hasattr(obj, attr):
            return getattr(obj, attr), parameter
    alias_names = ", ".join(attr for attr, _ in aliases)
    raise ValidationError(
        f"TBP cut objects at '{prefix}' must expose one of: {alias_names}.",
        parameter=prefix,
        value=type(obj).__name__,
    )


def _get_optional_object_attr(obj: object, *attrs: str) -> Any | None:
    for attr in attrs:
        if hasattr(obj, attr):
            return getattr(obj, attr)
    return None


def _get_optional_object_temperature_alias(obj: object, prefix: str) -> float | None:
    for attr, unit in (
        ("tb_k", "k"),
        ("boiling_point_k", "k"),
        ("tb_c", "c"),
        ("boiling_point_c", "c"),
        ("tb_f", "f"),
        ("boiling_point_f", "f"),
    ):
        if not hasattr(obj, attr):
            continue
        value = getattr(obj, attr)
        if value is None:
            continue
        if unit == "k":
            return _as_positive_float(value, f"{prefix}.{attr}", "TBP cut boiling point must be positive.")
        if unit == "c":
            return _as_positive_float(value, f"{prefix}.{attr}", "TBP cut boiling point must be positive.") + 273.15
        return (
            _as_positive_float(value, f"{prefix}.{attr}", "TBP cut boiling point must be positive.") - 32.0
        ) * (5.0 / 9.0) + 273.15
    return None


def _as_non_empty_str(value: Any, parameter: str) -> str:
    if not isinstance(value, str):
        raise ValidationError(
            f"Expected string at '{parameter}'.",
            parameter=parameter,
            value=value,
        )
    normalized = value.strip()
    if not normalized:
        raise ValidationError(
            "TBP cut names must be non-empty strings.",
            parameter=parameter,
            value=value,
        )
    return normalized


def _as_positive_int(value: Any, parameter: str) -> int:
    if isinstance(value, bool):
        raise ValidationError(
            f"Expected integer at '{parameter}'.",
            parameter=parameter,
            value=value,
        )
    try:
        converted = int(value)
    except (TypeError, ValueError):
        raise ValidationError(
            f"Expected integer at '{parameter}'.",
            parameter=parameter,
            value=value,
        )
    if converted <= 0:
        raise ValidationError(
            "TBP cut carbon numbers must be positive integers.",
            parameter=parameter,
            value=converted,
        )
    return converted


def _as_positive_float(value: Any, parameter: str, message: str) -> float:
    if isinstance(value, bool):
        raise ValidationError(message, parameter=parameter, value=value)
    try:
        converted = float(value)
    except (TypeError, ValueError):
        raise ValidationError(message, parameter=parameter, value=value)
    if not isfinite(converted) or converted <= 0.0:
        raise ValidationError(message, parameter=parameter, value=converted)
    return converted


__all__ = [
    "TBPAssayCut",
    "TBPCutResult",
    "TBPResult",
    "simulate_tbp",
]

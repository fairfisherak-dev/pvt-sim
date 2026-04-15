import numpy as np
import pytest

from pvtcore.characterization.plus_splitting.pedersen import (
    PedersenTBPCutConstraint,
    split_plus_fraction_pedersen,
)


def test_pedersen_split_closes_mass_and_mw_balance() -> None:
    z_plus = 0.25
    MW_plus = 215.0
    res = split_plus_fraction_pedersen(z_plus=z_plus, MW_plus=MW_plus, n_start=7, n_end=45)

    assert res.z.shape == res.MW.shape == res.n.shape
    assert np.all(res.z > 0.0)

    s1 = float(res.z.sum())
    s2 = float((res.z * res.MW).sum())

    assert s1 == pytest.approx(z_plus, abs=1e-10)
    assert s2 == pytest.approx(z_plus * MW_plus, abs=1e-8)


def test_pedersen_split_rejects_invalid_inputs() -> None:
    with pytest.raises(ValueError):
        split_plus_fraction_pedersen(z_plus=0.0, MW_plus=200.0)
    with pytest.raises(ValueError):
        split_plus_fraction_pedersen(z_plus=0.1, MW_plus=0.0)
    with pytest.raises(ValueError):
        split_plus_fraction_pedersen(z_plus=0.1, MW_plus=200.0, n_start=10, n_end=7)


def test_pedersen_split_supports_fit_to_tbp_mode() -> None:
    tbp_cuts = (
        PedersenTBPCutConstraint(name="C7", carbon_number=7, carbon_number_end=7, z=0.020, mw=96.0),
        PedersenTBPCutConstraint(name="C8", carbon_number=8, carbon_number_end=8, z=0.015, mw=110.0),
        PedersenTBPCutConstraint(name="C9", carbon_number=9, carbon_number_end=9, z=0.015, mw=124.0),
    )

    res = split_plus_fraction_pedersen(
        z_plus=0.05,
        MW_plus=108.6,
        n_start=7,
        n_end=12,
        solve_ab_from="fit_to_tbp",
        tbp_cuts=tbp_cuts,
    )

    assert res.solve_ab_from == "fit_to_tbp"
    assert res.tbp_cut_rms_relative_error is not None
    assert res.tbp_cut_rms_relative_error < 0.35
    assert float(res.z.sum()) == pytest.approx(0.05, abs=1e-10)

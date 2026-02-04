import numpy as np
import pytest

from pvtcore.characterization.plus_splitting.pedersen import split_plus_fraction_pedersen


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

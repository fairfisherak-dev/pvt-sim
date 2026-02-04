import numpy as np

from pvtcore.characterization.scn_properties import get_scn_properties
from pvtcore.characterization.scn_tables.katz_firoozabadi_1978 import get_katz_firoozabadi_table


def test_katz_table_spot_checks():
    t = get_katz_firoozabadi_table()

    r7 = t[7]
    assert r7.mw == 96.0
    assert np.isclose(r7.sg_6060, 0.722)
    assert np.isclose(r7.tb_c, 91.9)

    r20 = t[20]
    assert r20.mw == 275.0
    assert np.isclose(r20.sg_6060, 0.862)
    assert np.isclose(r20.tb_f, 640.5)

    r45 = t[45]
    assert r45.mw == 626.0
    assert np.isclose(r45.sg_6060, 0.937)
    assert np.isclose(r45.tb_c, 553.0)


def test_get_scn_properties_table_range():
    props = get_scn_properties(n_start=7, n_end=45, extrapolate=False)
    assert props.n[0] == 7
    assert props.n[-1] == 45
    assert props.mw.shape == props.n.shape
    assert np.all(np.diff(props.tb_k) > 0.0)  # monotone increasing in the table


def test_get_scn_properties_extrapolates_beyond_45():
    props = get_scn_properties(n_start=43, n_end=50, extrapolate=True)
    assert props.n[0] == 43
    assert props.n[-1] == 50
    # C45 should come from table exactly
    idx45 = int(np.where(props.n == 45)[0][0])
    assert np.isclose(props.sg_6060[idx45], 0.937)
    assert np.isclose(props.mw[idx45], 626.0)
    # Beyond C45 should be finite and plausible
    idx50 = int(np.where(props.n == 50)[0][0])
    assert props.mw[idx50] > props.mw[idx45]
    assert props.sg_6060[idx50] > 0.8
    assert props.tb_k[idx50] > props.tb_k[idx45]

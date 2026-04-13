"""Unit tests for the Peng-Robinson (1978) EOS variant."""

import numpy as np
import pytest

from pvtcore.eos.peng_robinson import PengRobinsonEOS
from pvtcore.eos.pr78 import PR78EOS
from pvtcore.models.component import load_components


@pytest.fixture
def components():
    return load_components()


@pytest.fixture
def c12_pr76(components):
    return PengRobinsonEOS([components["C12"]])


@pytest.fixture
def c12_pr78(components):
    return PR78EOS([components["C12"]])


def test_pr78_uses_extended_kappa_for_heavy_component(c12_pr78, components):
    omega = components["C12"].omega
    assert omega > 0.49
    expected = (
        0.379642
        + 1.48503 * omega
        - 0.164423 * omega ** 2
        + 0.016666 * omega ** 3
    )
    assert c12_pr78.kappa[0] == pytest.approx(expected, rel=1e-10)


def test_pr78_differs_from_pr76_for_heavy_component(c12_pr76, c12_pr78):
    assert c12_pr78.kappa[0] != pytest.approx(c12_pr76.kappa[0], rel=1e-12)


def test_pr78_changes_compressibility_for_heavy_component(c12_pr76, c12_pr78):
    composition = np.array([1.0])
    z_pr76 = c12_pr76.compressibility(3.0e6, 650.0, composition, phase="vapor")
    z_pr78 = c12_pr78.compressibility(3.0e6, 650.0, composition, phase="vapor")
    assert z_pr78 != pytest.approx(z_pr76, rel=1e-10)


def test_pr78_binary_fugacity_coefficients_are_positive(components):
    eos = PR78EOS([components["C1"], components["C12"]])
    composition = np.array([0.5, 0.5])
    phi = eos.fugacity_coefficient(5.0e6, 400.0, composition, phase="vapor")
    assert phi.shape == (2,)
    assert np.all(phi > 0.0)

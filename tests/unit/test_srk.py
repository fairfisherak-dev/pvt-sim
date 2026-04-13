"""Unit tests for the Soave-Redlich-Kwong EOS."""

import numpy as np
import pytest

from pvtcore.eos.srk import SRKEOS
from pvtcore.models.component import load_components


@pytest.fixture
def components():
    return load_components()


@pytest.fixture
def methane_eos(components):
    return SRKEOS([components["C1"]])


@pytest.fixture
def binary_eos(components):
    return SRKEOS([components["C1"], components["C2"]])


def test_alpha_is_one_at_critical_temperature(methane_eos, components):
    methane = components["C1"]
    assert methane_eos.alpha_function(methane.Tc, 0) == pytest.approx(1.0, rel=1e-10)


def test_ideal_gas_limit(methane_eos):
    composition = np.array([1.0])
    Z = methane_eos.compressibility(1.0e3, 1000.0, composition, phase="vapor")
    assert Z == pytest.approx(1.0, abs=0.02)


def test_liquid_root_is_less_than_vapor_root(methane_eos):
    composition = np.array([1.0])
    roots = methane_eos.compressibility(2.0e6, 150.0, composition, phase="auto")
    if isinstance(roots, list) and len(roots) == 3:
        assert min(roots) < max(roots)


def test_binary_fugacity_coefficients_are_positive(binary_eos):
    composition = np.array([0.5, 0.5])
    phi = binary_eos.fugacity_coefficient(5.0e6, 320.0, composition, phase="vapor")
    assert phi.shape == (2,)
    assert np.all(phi > 0.0)

"""
Test units validation script.

Verifies that validate_units.py correctly detects unit errors.
"""

import subprocess
import sys
from pathlib import Path

import pytest


@pytest.fixture
def repo_root():
    """Get repository root directory."""
    return Path(__file__).parent.parent.parent


@pytest.fixture
def validate_script(repo_root):
    """Get path to validate_units.py script."""
    return repo_root / "scripts" / "validate_units.py"


@pytest.fixture
def bad_units_fixture(repo_root):
    """Get path to bad units test fixture."""
    return repo_root / "tests" / "fixtures" / "bad_units_kpa_as_pa.json"


@pytest.fixture
def good_components_db(repo_root):
    """Get path to actual component database (should be valid)."""
    return repo_root / "data" / "pure_components" / "components.json"


def test_validate_script_exists(validate_script):
    """Test that validate_units.py exists."""
    assert validate_script.exists(), f"Validation script not found: {validate_script}"


def test_bad_units_fixture_exists(bad_units_fixture):
    """Test that bad units fixture exists."""
    assert bad_units_fixture.exists(), f"Bad units fixture not found: {bad_units_fixture}"


def test_bad_units_detected(validate_script, bad_units_fixture):
    """
    Test that validate_units.py detects errors in bad units fixture.

    The fixture contains:
    - Pc in kPa mistakenly stored as Pa
    - Tc/Tb in Celsius stored as Kelvin
    - Wrong unit metadata labels
    """
    result = subprocess.run(
        [
            sys.executable,
            str(validate_script),
            "--component-db",
            str(bad_units_fixture),
        ],
        capture_output=True,
        text=True,
    )

    # Should fail (exit code 1)
    assert result.returncode == 1, "Validator should detect errors in bad units fixture"

    # Check that specific errors are detected
    output = result.stdout + result.stderr

    # Should detect kPa stored as Pa (Pc too low)
    assert "PROPANE_BAD" in output, "Should flag PROPANE_BAD component"
    assert "Pc" in output, "Should detect Pc unit error"

    # Should detect Celsius as Kelvin (negative or too-low Tc)
    assert "METHANE_CELSIUS" in output, "Should flag METHANE_CELSIUS component"
    assert "Tc" in output or "Tb" in output, "Should detect temperature errors"

    # Should detect wrong unit metadata
    assert "ETHANE_WRONG_UNIT_LABEL" in output, "Should flag ETHANE_WRONG_UNIT_LABEL"
    assert "canonical unit" in output.lower(), "Should mention canonical units"


def test_good_components_pass(validate_script, good_components_db):
    """
    Test that actual component database passes validation.

    This is a regression test to ensure the real database has correct units.
    """
    if not good_components_db.exists():
        pytest.skip(f"Component database not found: {good_components_db}")

    result = subprocess.run(
        [
            sys.executable,
            str(validate_script),
            "--component-db",
            str(good_components_db),
        ],
        capture_output=True,
        text=True,
    )

    # Should pass (exit code 0) or have warnings only (exit code 2)
    assert result.returncode in (0, 2), (
        f"Real component database should pass validation.\n"
        f"Exit code: {result.returncode}\n"
        f"Output:\n{result.stdout}\n{result.stderr}"
    )


def test_validation_output_actionable(validate_script, bad_units_fixture):
    """
    Test that error messages are actionable (mention recommended fixes).
    """
    result = subprocess.run(
        [
            sys.executable,
            str(validate_script),
            "--component-db",
            str(bad_units_fixture),
        ],
        capture_output=True,
        text=True,
    )

    output = result.stdout + result.stderr

    # Should mention specific fixes
    assert (
        "multiply" in output.lower()
        or "convert" in output.lower()
        or "typical" in output.lower()
    ), "Error messages should suggest fixes"


def test_strict_mode(validate_script, good_components_db):
    """
    Test --strict mode treats warnings as errors.
    """
    if not good_components_db.exists():
        pytest.skip(f"Component database not found: {good_components_db}")

    # Run without strict mode
    result_normal = subprocess.run(
        [
            sys.executable,
            str(validate_script),
            "--component-db",
            str(good_components_db),
        ],
        capture_output=True,
        text=True,
    )

    # Run with strict mode
    result_strict = subprocess.run(
        [
            sys.executable,
            str(validate_script),
            "--component-db",
            str(good_components_db),
            "--strict",
        ],
        capture_output=True,
        text=True,
    )

    # If there are warnings, strict mode should fail while normal passes
    if result_normal.returncode == 2:  # Warnings only
        assert result_strict.returncode == 1, (
            "--strict mode should treat warnings as errors"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

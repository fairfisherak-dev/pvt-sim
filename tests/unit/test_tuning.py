"""Unit tests for the tuning/regression module."""

import numpy as np
import pytest

from pvtcore.tuning import (
    # Data types
    DataType,
    ExperimentalPoint,
    ExperimentalDataSet,
    ObjectiveResult,
    ObjectiveFunction,
    # Objective helpers
    calculate_residual,
    calculate_objective_sse,
    calculate_objective_aad,
    create_saturation_objective,
    create_density_objective,
    # Parameter types
    ParameterType,
    TunableParameter,
    ParameterSet,
    # Parameter helpers
    create_kij_parameters,
    create_volume_shift_parameters,
    create_critical_multipliers,
    merge_parameter_sets,
    # Regression
    RegressionResult,
    EOSRegressor,
    tune_binary_interactions,
    sensitivity_analysis,
)
from pvtcore.models.component import load_components
from pvtcore.eos.peng_robinson import PengRobinsonEOS
from pvtcore.core.errors import ValidationError


@pytest.fixture
def components():
    """Load component database."""
    return load_components()


@pytest.fixture
def methane_propane(components):
    """Binary C1-C3 mixture components."""
    return [components['C1'], components['C3']]


@pytest.fixture
def methane_propane_eos(methane_propane):
    """EOS for C1-C3 binary."""
    return PengRobinsonEOS(methane_propane)


# =============================================================================
# Objective Function Tests
# =============================================================================

class TestResiduals:
    """Tests for residual calculations."""

    def test_calculate_residual_relative(self):
        """Test relative residual calculation."""
        calc = 105.0
        exp = 100.0
        residual = calculate_residual(calc, exp, relative=True)
        assert abs(residual - 0.05) < 1e-10

    def test_calculate_residual_absolute(self):
        """Test absolute residual calculation."""
        calc = 105.0
        exp = 100.0
        residual = calculate_residual(calc, exp, relative=False)
        assert abs(residual - 5.0) < 1e-10

    def test_calculate_residual_zero_experimental(self):
        """Test residual when experimental is zero."""
        residual = calculate_residual(5.0, 0.0, relative=True)
        assert residual == 5.0

    def test_calculate_residual_both_zero(self):
        """Test residual when both are zero."""
        residual = calculate_residual(0.0, 0.0, relative=True)
        assert residual == 0.0


class TestObjectiveMetrics:
    """Tests for objective function metrics."""

    def test_sse(self):
        """Test sum of squared errors."""
        residuals = np.array([0.1, -0.2, 0.15])
        sse = calculate_objective_sse(residuals)
        expected = 0.1**2 + 0.2**2 + 0.15**2
        assert abs(sse - expected) < 1e-10

    def test_aad(self):
        """Test average absolute deviation."""
        residuals = np.array([0.1, -0.2, 0.1])
        aad = calculate_objective_aad(residuals)
        # AAD returns percentage
        expected = np.mean([0.1, 0.2, 0.1]) * 100
        assert abs(aad - expected) < 1e-10


class TestExperimentalPoint:
    """Tests for ExperimentalPoint class."""

    def test_create_point(self):
        """Test creating an experimental point."""
        point = ExperimentalPoint(
            data_type=DataType.SATURATION_PRESSURE,
            temperature=300.0,
            pressure=None,
            value=5e6,
        )
        assert point.data_type == DataType.SATURATION_PRESSURE
        assert point.temperature == 300.0
        assert point.value == 5e6
        assert point.weight == 1.0

    def test_point_with_uncertainty(self):
        """Test point with uncertainty."""
        point = ExperimentalPoint(
            data_type=DataType.LIQUID_DENSITY,
            temperature=300.0,
            pressure=5e6,
            value=500.0,
            uncertainty=5.0,
        )
        assert point.uncertainty == 5.0


class TestExperimentalDataSet:
    """Tests for ExperimentalDataSet class."""

    def test_create_dataset(self):
        """Test creating a dataset."""
        z = np.array([0.5, 0.5])
        points = [
            ExperimentalPoint(DataType.SATURATION_PRESSURE, 300.0, None, 5e6),
            ExperimentalPoint(DataType.SATURATION_PRESSURE, 320.0, None, 7e6),
        ]
        dataset = ExperimentalDataSet("test", z, points)

        assert dataset.n_points == 2
        assert len(dataset.data_types) == 1

    def test_dataset_normalizes_composition(self):
        """Test that composition is normalized."""
        z = np.array([1.0, 1.0])  # Sum = 2
        points = [ExperimentalPoint(DataType.SATURATION_PRESSURE, 300.0, None, 5e6)]
        dataset = ExperimentalDataSet("test", z, points)

        assert abs(dataset.composition.sum() - 1.0) < 1e-10

    def test_dataset_requires_points(self):
        """Test that empty points raises error."""
        z = np.array([0.5, 0.5])
        with pytest.raises(ValidationError):
            ExperimentalDataSet("test", z, [])

    def test_get_points_by_type(self):
        """Test filtering points by type."""
        z = np.array([0.5, 0.5])
        points = [
            ExperimentalPoint(DataType.SATURATION_PRESSURE, 300.0, None, 5e6),
            ExperimentalPoint(DataType.LIQUID_DENSITY, 300.0, 5e6, 500.0),
        ]
        dataset = ExperimentalDataSet("test", z, points)

        sat_points = dataset.get_points_by_type(DataType.SATURATION_PRESSURE)
        assert len(sat_points) == 1


class TestCreateObjectives:
    """Tests for objective creation helpers."""

    def test_create_saturation_objective(self):
        """Test creating saturation pressure dataset."""
        T = np.array([300.0, 320.0, 340.0])
        P = np.array([5e6, 7e6, 9e6])
        z = np.array([0.5, 0.5])

        dataset = create_saturation_objective(T, P, z, 'bubble')

        assert dataset.n_points == 3
        assert dataset.name == "bubble_pressure"

    def test_create_density_objective(self):
        """Test creating density dataset."""
        T = np.array([300.0, 320.0])
        P = np.array([5e6, 10e6])
        rho = np.array([500.0, 550.0])
        z = np.array([0.5, 0.5])

        dataset = create_density_objective(T, P, rho, z, 'liquid')

        assert dataset.n_points == 2
        assert dataset.name == "liquid_density"


# =============================================================================
# Parameter Tests
# =============================================================================

class TestTunableParameter:
    """Tests for TunableParameter class."""

    def test_create_parameter(self):
        """Test creating a tunable parameter."""
        param = TunableParameter(
            name="kij_C1_C3",
            param_type=ParameterType.BINARY_INTERACTION,
            initial_value=0.0,
            lower_bound=-0.1,
            upper_bound=0.1,
        )
        assert param.name == "kij_C1_C3"
        assert param.bounds == (-0.1, 0.1)
        assert param.active is True

    def test_parameter_bounds_validation(self):
        """Test that invalid bounds raise error."""
        with pytest.raises(ValidationError):
            TunableParameter(
                name="test",
                param_type=ParameterType.BINARY_INTERACTION,
                initial_value=0.0,
                lower_bound=0.1,  # Lower > upper
                upper_bound=-0.1,
            )

    def test_parameter_initial_value_validation(self):
        """Test that initial value outside bounds raises error."""
        with pytest.raises(ValidationError):
            TunableParameter(
                name="test",
                param_type=ParameterType.BINARY_INTERACTION,
                initial_value=0.5,  # Outside bounds
                lower_bound=-0.1,
                upper_bound=0.1,
            )


class TestParameterSet:
    """Tests for ParameterSet class."""

    def test_create_parameter_set(self):
        """Test creating a parameter set."""
        param_set = ParameterSet(n_components=2)
        param_set.add_parameter(TunableParameter(
            name="kij_0_1",
            param_type=ParameterType.BINARY_INTERACTION,
            initial_value=0.0,
            lower_bound=-0.1,
            upper_bound=0.1,
        ))
        assert len(param_set.parameters) == 1

    def test_duplicate_parameter_error(self):
        """Test that duplicate names raise error."""
        param_set = ParameterSet(n_components=2)
        param = TunableParameter(
            name="test",
            param_type=ParameterType.BINARY_INTERACTION,
            initial_value=0.0,
            lower_bound=-0.1,
            upper_bound=0.1,
        )
        param_set.add_parameter(param)

        with pytest.raises(ValidationError):
            param_set.add_parameter(param)

    def test_get_initial_vector(self):
        """Test getting initial values as vector."""
        param_set = ParameterSet(n_components=2)
        param_set.add_parameter(TunableParameter(
            name="p1", param_type=ParameterType.BINARY_INTERACTION,
            initial_value=0.05, lower_bound=-0.1, upper_bound=0.1,
        ))
        param_set.add_parameter(TunableParameter(
            name="p2", param_type=ParameterType.BINARY_INTERACTION,
            initial_value=-0.02, lower_bound=-0.1, upper_bound=0.1,
        ))

        x0 = param_set.get_initial_vector()
        assert len(x0) == 2
        assert x0[0] == 0.05
        assert x0[1] == -0.02

    def test_vector_to_dict(self):
        """Test converting vector to dictionary."""
        param_set = ParameterSet(n_components=2)
        param_set.add_parameter(TunableParameter(
            name="p1", param_type=ParameterType.BINARY_INTERACTION,
            initial_value=0.0, lower_bound=-0.1, upper_bound=0.1,
        ))
        param_set.add_parameter(TunableParameter(
            name="p2", param_type=ParameterType.BINARY_INTERACTION,
            initial_value=0.0, lower_bound=-0.1, upper_bound=0.1,
        ))

        x = np.array([0.03, -0.01])
        params = param_set.vector_to_dict(x)

        assert params["p1"] == 0.03
        assert params["p2"] == -0.01


class TestParameterCreationHelpers:
    """Tests for parameter creation helpers."""

    def test_create_kij_parameters(self):
        """Test creating kij parameter set."""
        params = create_kij_parameters(3, ['C1', 'C3', 'C7'])

        # 3 components -> 3 unique pairs
        assert params.n_active == 3
        assert params.n_components == 3

    def test_create_kij_with_active_pairs(self):
        """Test creating kij with specific active pairs."""
        params = create_kij_parameters(
            3, ['C1', 'C3', 'C7'],
            active_pairs=[(0, 2)]  # Only C1-C7
        )

        assert params.n_active == 1

    def test_create_volume_shift_parameters(self):
        """Test creating volume shift parameters."""
        params = create_volume_shift_parameters(3, ['C1', 'C3', 'C7'])

        assert params.n_active == 3

    def test_merge_parameter_sets(self):
        """Test merging parameter sets."""
        kij_params = create_kij_parameters(2, ['C1', 'C3'])
        shift_params = create_volume_shift_parameters(2, ['C1', 'C3'])

        merged = merge_parameter_sets(kij_params, shift_params)

        # 1 kij pair + 2 shifts = 3 parameters
        assert len(merged.parameters) == 3


class TestExtractParameters:
    """Tests for extracting parameters from dict."""

    def test_extract_kij_matrix(self):
        """Test extracting kij matrix."""
        params = create_kij_parameters(3, ['C1', 'C3', 'C7'])

        param_dict = {
            'kij_C1_C3': 0.01,
            'kij_C1_C7': 0.02,
            'kij_C3_C7': 0.005,
        }

        kij = params.extract_kij_matrix(param_dict)

        assert kij[0, 1] == 0.01
        assert kij[1, 0] == 0.01  # Symmetric
        assert kij[0, 2] == 0.02
        assert kij[1, 2] == 0.005

    def test_extract_volume_shifts(self):
        """Test extracting volume shifts."""
        params = create_volume_shift_parameters(2, ['C1', 'C3'])

        param_dict = {
            'c_C1': 0.001,
            'c_C3': -0.002,
        }

        shifts = params.extract_volume_shifts(param_dict)

        assert shifts[0] == 0.001
        assert shifts[1] == -0.002


# =============================================================================
# Regression Tests
# =============================================================================

class TestEOSRegressor:
    """Tests for EOS regression engine."""

    def test_regressor_init(self, methane_propane, methane_propane_eos):
        """Test creating a regressor."""
        params = create_kij_parameters(2, ['C1', 'C3'])
        regressor = EOSRegressor(methane_propane, methane_propane_eos, params)

        assert regressor.components == methane_propane
        assert len(regressor.datasets) == 0

    def test_add_dataset(self, methane_propane, methane_propane_eos):
        """Test adding a dataset."""
        params = create_kij_parameters(2, ['C1', 'C3'])
        regressor = EOSRegressor(methane_propane, methane_propane_eos, params)

        z = np.array([0.5, 0.5])
        data = create_saturation_objective(
            np.array([300.0]), np.array([5e6]), z, 'bubble'
        )
        regressor.add_dataset(data)

        assert len(regressor.datasets) == 1

    def test_regressor_no_datasets_error(self, methane_propane, methane_propane_eos):
        """Test that fit without datasets raises error."""
        params = create_kij_parameters(2, ['C1', 'C3'])
        regressor = EOSRegressor(methane_propane, methane_propane_eos, params)

        with pytest.raises(ValidationError):
            regressor.fit()

    def test_regressor_no_active_params_error(self, methane_propane, methane_propane_eos):
        """Test that fit without active parameters raises error."""
        params = create_kij_parameters(2, ['C1', 'C3'], active_pairs=[])  # No active
        regressor = EOSRegressor(methane_propane, methane_propane_eos, params)

        z = np.array([0.5, 0.5])
        data = create_saturation_objective(
            np.array([300.0]), np.array([5e6]), z, 'bubble'
        )
        regressor.add_dataset(data)

        with pytest.raises(ValidationError):
            regressor.fit()


class TestRegressionResult:
    """Tests for RegressionResult class."""

    def test_result_structure(self):
        """Test result dataclass structure."""
        result = RegressionResult(
            success=True,
            optimal_params={'kij': 0.01},
            initial_objective=1.0,
            final_objective=0.1,
            improvement=90.0,
            n_iterations=10,
            n_evaluations=50,
            elapsed_time=1.5,
            convergence_message="Converged",
            parameter_set=ParameterSet(),
        )

        assert result.success is True
        assert result.improvement == 90.0


# =============================================================================
# Integration Tests
# =============================================================================

class TestTuningIntegration:
    """Integration tests for the full tuning workflow."""

    def test_simple_regression(self, methane_propane, methane_propane_eos):
        """Test a simple regression case.

        Note: This uses synthetic data so we know the expected result.
        """
        # Create synthetic experimental data
        # Use model to generate "experimental" data with kij=0.01
        z = np.array([0.5, 0.5])

        # Create data points
        T_exp = np.array([280.0, 300.0, 320.0])
        P_exp = np.array([2.5e6, 4.0e6, 5.5e6])  # Approximate values

        data = create_saturation_objective(T_exp, P_exp, z, 'bubble')

        # Create regressor with kij starting at 0
        params = create_kij_parameters(2, ['C1', 'C3'])
        regressor = EOSRegressor(methane_propane, methane_propane_eos, params)
        regressor.add_dataset(data)

        # Run with minimal iterations (just test it runs)
        result = regressor.fit(method='Nelder-Mead', maxiter=5)

        # Check result structure
        assert isinstance(result, RegressionResult)
        assert 'kij_C1_C3' in result.optimal_params
        assert result.n_evaluations > 0

    def test_objective_function_standalone(self, methane_propane, methane_propane_eos):
        """Test ObjectiveFunction class independently."""
        z = np.array([0.5, 0.5])

        # Create dataset with a single point
        points = [ExperimentalPoint(
            data_type=DataType.VAPOR_FRACTION,
            temperature=300.0,
            pressure=3e6,
            value=0.5,  # Expected vapor fraction
        )]
        dataset = ExperimentalDataSet("test", z, points)

        # Model function that returns fixed value
        def mock_model(point, params):
            return 0.55  # Slightly off

        obj_func = ObjectiveFunction([dataset], mock_model, metric='sse')
        value = obj_func({})

        # Should have some positive error
        assert value > 0

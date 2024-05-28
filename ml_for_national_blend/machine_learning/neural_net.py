"""Helper methods for training a neural network."""

import os
import pickle
import warnings
import numpy
import keras
from tensorflow.keras.saving import load_model
from gewittergefahr.gg_utils import time_conversion
from gewittergefahr.gg_utils import time_periods
from gewittergefahr.gg_utils import temperature_conversions as temperature_conv
from gewittergefahr.gg_utils import file_system_utils
from gewittergefahr.gg_utils import error_checking
from ml_for_national_blend.io import nwp_model_io
from ml_for_national_blend.io import interp_nwp_model_io
from ml_for_national_blend.io import nbm_constant_io
from ml_for_national_blend.io import urma_io
from ml_for_national_blend.utils import nwp_model_utils
from ml_for_national_blend.utils import urma_utils
from ml_for_national_blend.utils import nbm_constant_utils
from ml_for_national_blend.utils import normalization
from ml_for_national_blend.machine_learning import custom_losses
from ml_for_national_blend.machine_learning import custom_metrics

TIME_FORMAT = '%Y-%m-%d-%H'
HOURS_TO_SECONDS = 3600
DEFAULT_GUST_FACTOR = 1.5

FIRST_INIT_TIMES_KEY = 'first_init_times_unix_sec'
LAST_INIT_TIMES_KEY = 'last_init_times_unix_sec'
NWP_LEAD_TIMES_KEY = 'nwp_lead_times_hours'
NWP_MODEL_TO_DIR_KEY = 'nwp_model_to_dir_name'
NWP_MODEL_TO_FIELDS_KEY = 'nwp_model_to_field_names'
NWP_NORM_FILE_KEY = 'nwp_normalization_file_name'
NWP_USE_QUANTILE_NORM_KEY = 'nwp_use_quantile_norm'
TARGET_LEAD_TIME_KEY = 'target_lead_time_hours'
TARGET_FIELDS_KEY = 'target_field_names'
TARGET_DIR_KEY = 'target_dir_name'
TARGET_NORM_FILE_KEY = 'target_normalization_file_name'
TARGETS_USE_QUANTILE_NORM_KEY = 'targets_use_quantile_norm'
NBM_CONSTANT_FIELDS_KEY = 'nbm_constant_field_names'
NBM_CONSTANT_FILE_KEY = 'nbm_constant_file_name'
BATCH_SIZE_KEY = 'num_examples_per_batch'
SENTINEL_VALUE_KEY = 'sentinel_value'
SUBSET_GRID_KEY = 'subset_grid'

PREDICT_DEWPOINT_DEPRESSION_KEY = 'predict_dewpoint_depression'
PREDICT_GUST_FACTOR_KEY = 'predict_gust_factor'
DO_RESIDUAL_PREDICTION_KEY = 'do_residual_prediction'
RESID_BASELINE_MODEL_KEY = 'resid_baseline_model_name'
RESID_BASELINE_LEAD_TIME_KEY = 'resid_baseline_lead_time_hours'
RESID_BASELINE_MODEL_DIR_KEY = 'resid_baseline_model_dir_name'

DEFAULT_GENERATOR_OPTION_DICT = {
    SENTINEL_VALUE_KEY: -10.
}

PREDICTOR_MATRICES_KEY = 'predictor_matrices_key'
TARGET_MATRIX_KEY = 'target_matrix'
INIT_TIMES_KEY = 'init_times_unix_sec'

NUM_EPOCHS_KEY = 'num_epochs'
NUM_TRAINING_BATCHES_KEY = 'num_training_batches_per_epoch'
TRAINING_OPTIONS_KEY = 'training_option_dict'
NUM_VALIDATION_BATCHES_KEY = 'num_validation_batches_per_epoch'
VALIDATION_OPTIONS_KEY = 'validation_option_dict'
LOSS_FUNCTION_KEY = 'loss_function_string'
OPTIMIZER_FUNCTION_KEY = 'optimizer_function_string'
METRIC_FUNCTIONS_KEY = 'metric_function_strings'
PLATEAU_PATIENCE_KEY = 'plateau_patience_epochs'
PLATEAU_LR_MUTIPLIER_KEY = 'plateau_learning_rate_multiplier'
EARLY_STOPPING_PATIENCE_KEY = 'early_stopping_patience_epochs'

METADATA_KEYS = [
    NUM_EPOCHS_KEY, NUM_TRAINING_BATCHES_KEY, TRAINING_OPTIONS_KEY,
    NUM_VALIDATION_BATCHES_KEY, VALIDATION_OPTIONS_KEY, LOSS_FUNCTION_KEY,
    OPTIMIZER_FUNCTION_KEY, METRIC_FUNCTIONS_KEY, PLATEAU_PATIENCE_KEY,
    PLATEAU_LR_MUTIPLIER_KEY, EARLY_STOPPING_PATIENCE_KEY
]


def __nwp_2m_dewpoint_to_depression(nwp_forecast_table_xarray):
    """Converts 2-metre NWP dewpoints to dewpoint depressions.

    :param nwp_forecast_table_xarray: xarray table in format returned by
        `interp_nwp_model_io.read_file`.
    :return: nwp_forecast_table_xarray: Same, except that dewpoints have been
        replaced with dewpoint depressions.
    """

    dewpoint_matrix_kelvins = nwp_model_utils.get_field(
        nwp_forecast_table_xarray=nwp_forecast_table_xarray,
        field_name=nwp_model_utils.DEWPOINT_2METRE_NAME
    )
    temperature_matrix_kelvins = nwp_model_utils.get_field(
        nwp_forecast_table_xarray=nwp_forecast_table_xarray,
        field_name=nwp_model_utils.TEMPERATURE_2METRE_NAME
    )
    dewp_depression_matrix_kelvins = (
        temperature_matrix_kelvins - dewpoint_matrix_kelvins
    )

    k = numpy.where(
        nwp_forecast_table_xarray.coords[nwp_model_utils.FIELD_DIM].values ==
        nwp_model_utils.DEWPOINT_2METRE_NAME
    )[0][0]

    data_matrix = nwp_forecast_table_xarray[nwp_model_utils.DATA_KEY].values
    data_matrix[..., k] = dewp_depression_matrix_kelvins

    return nwp_forecast_table_xarray.assign({
        nwp_model_utils.DATA_KEY: (
            nwp_forecast_table_xarray[nwp_model_utils.DATA_KEY].dims,
            data_matrix
        )
    })


def __nwp_2m_temp_to_celsius(nwp_forecast_table_xarray):
    """Converts 2-metre NWP temperatures from Kelvins to Celsius.

    :param nwp_forecast_table_xarray: xarray table in format returned by
        `interp_nwp_model_io.read_file`.
    :return: nwp_forecast_table_xarray: Same, except that temperatures have been
        converted to Celsius.
    """

    k = numpy.where(
        nwp_forecast_table_xarray.coords[nwp_model_utils.FIELD_DIM].values ==
        nwp_model_utils.TEMPERATURE_2METRE_NAME
    )[0][0]

    data_matrix = nwp_forecast_table_xarray[nwp_model_utils.DATA_KEY].values
    data_matrix[..., k] = temperature_conv.kelvins_to_celsius(
        data_matrix[..., k]
    )

    return nwp_forecast_table_xarray.assign({
        nwp_model_utils.DATA_KEY: (
            nwp_forecast_table_xarray[nwp_model_utils.DATA_KEY].dims,
            data_matrix
        )
    })


def __predicted_2m_depression_to_dewpoint(prediction_matrix,
                                          target_field_names):
    """Converts 2-metre dewpoint depression in predictions to dewpoint temp.

    :param prediction_matrix: See documentation for `apply_model`.
    :param target_field_names: Same.
    :return: prediction_matrix: Same as input, except that dewpoint depressions
        have been replaced with dewpoint temperatures.
    """

    num_target_fields = prediction_matrix.shape[-1]
    target_field_names = numpy.array(target_field_names)

    error_checking.assert_is_numpy_array(
        target_field_names,
        exact_dimensions=numpy.array([num_target_fields], dtype=int)
    )

    dewp_index = numpy.where(
        target_field_names == urma_utils.DEWPOINT_2METRE_NAME
    )[0][0]
    temp_index = numpy.where(
        target_field_names == urma_utils.TEMPERATURE_2METRE_NAME
    )[0][0]
    prediction_matrix[..., dewp_index] = (
        prediction_matrix[..., temp_index] -
        prediction_matrix[..., dewp_index]
    )

    return prediction_matrix


def __nwp_10m_gust_speed_to_factor(nwp_forecast_table_xarray):
    """Converts 10-metre NWP gust speeds to gust factors.

    *** Actually, gust factors minus one. ***

    :param nwp_forecast_table_xarray: xarray table in format returned by
        `interp_nwp_model_io.read_file`.
    :return: nwp_forecast_table_xarray: Same, except that gust speeds have been
        replaced with gust factors.
    """

    k_inds = numpy.where(
        nwp_forecast_table_xarray.coords[nwp_model_utils.FIELD_DIM].values ==
        nwp_model_utils.WIND_GUST_10METRE_NAME
    )[0]

    if len(k_inds) == 0:
        return nwp_forecast_table_xarray

    u_wind_matrix_m_s01 = nwp_model_utils.get_field(
        nwp_forecast_table_xarray=nwp_forecast_table_xarray,
        field_name=nwp_model_utils.U_WIND_10METRE_NAME
    )
    v_wind_matrix_m_s01 = nwp_model_utils.get_field(
        nwp_forecast_table_xarray=nwp_forecast_table_xarray,
        field_name=nwp_model_utils.V_WIND_10METRE_NAME
    )
    sustained_speed_matrix_m_s01 = numpy.sqrt(
        u_wind_matrix_m_s01 ** 2 + v_wind_matrix_m_s01 ** 2
    )
    gust_speed_matrix_m_s01 = nwp_model_utils.get_field(
        nwp_forecast_table_xarray=nwp_forecast_table_xarray,
        field_name=nwp_model_utils.WIND_GUST_10METRE_NAME
    )

    data_matrix = nwp_forecast_table_xarray[nwp_model_utils.DATA_KEY].values
    k = k_inds[0]
    data_matrix[..., k] = (
        gust_speed_matrix_m_s01 / sustained_speed_matrix_m_s01 - 1.
    )

    return nwp_forecast_table_xarray.assign({
        nwp_model_utils.DATA_KEY: (
            nwp_forecast_table_xarray[nwp_model_utils.DATA_KEY].dims,
            data_matrix
        )
    })


def __predicted_10m_gust_factor_to_speed(prediction_matrix, target_field_names):
    """Converts 10-metre gust factor in predictions to gust speed.

    :param prediction_matrix: See documentation for `apply_model`.
    :param target_field_names: Same.
    :return: prediction_matrix: Same as input, except that gust factors have
        been replaced with gust speeds.
    """

    num_target_fields = prediction_matrix.shape[-1]
    target_field_names = numpy.array(target_field_names)

    error_checking.assert_is_numpy_array(
        target_field_names,
        exact_dimensions=numpy.array([num_target_fields], dtype=int)
    )

    u_index = numpy.where(
        target_field_names == urma_utils.U_WIND_10METRE_NAME
    )[0][0]
    v_index = numpy.where(
        target_field_names == urma_utils.V_WIND_10METRE_NAME
    )[0][0]
    gust_index = numpy.where(
        target_field_names == urma_utils.WIND_GUST_10METRE_NAME
    )[0][0]

    gust_factor_matrix = prediction_matrix[..., gust_index] + 1.
    sustained_speed_matrix = numpy.sqrt(
        prediction_matrix[..., u_index] ** 2 +
        prediction_matrix[..., v_index] ** 2
    )
    prediction_matrix[..., gust_index] = (
        gust_factor_matrix * sustained_speed_matrix
    )

    return prediction_matrix


def _check_generator_args(option_dict):
    """Checks input arguments for generator.

    :param option_dict: See doc for `data_generator`.
    :return: option_dict: Same as input, except defaults may have been added.
    """

    orig_option_dict = option_dict.copy()
    option_dict = DEFAULT_GENERATOR_OPTION_DICT.copy()
    option_dict.update(orig_option_dict)

    error_checking.assert_is_numpy_array(
        option_dict[FIRST_INIT_TIMES_KEY], num_dimensions=1
    )
    error_checking.assert_is_integer_numpy_array(
        option_dict[FIRST_INIT_TIMES_KEY]
    )

    expected_dim = numpy.array(
        [len(option_dict[FIRST_INIT_TIMES_KEY])], dtype=int
    )
    error_checking.assert_is_numpy_array(
        option_dict[LAST_INIT_TIMES_KEY], exact_dimensions=expected_dim
    )
    error_checking.assert_is_integer_numpy_array(
        option_dict[LAST_INIT_TIMES_KEY]
    )

    error_checking.assert_is_geq_numpy_array(
        option_dict[LAST_INIT_TIMES_KEY] - option_dict[FIRST_INIT_TIMES_KEY],
        0
    )

    error_checking.assert_is_numpy_array(
        option_dict[NWP_LEAD_TIMES_KEY], num_dimensions=1
    )
    error_checking.assert_is_integer_numpy_array(
        option_dict[NWP_LEAD_TIMES_KEY]
    )
    error_checking.assert_is_greater_numpy_array(
        option_dict[NWP_LEAD_TIMES_KEY], 0
    )
    option_dict[NWP_LEAD_TIMES_KEY] = numpy.unique(
        option_dict[NWP_LEAD_TIMES_KEY]
    )

    nwp_model_to_dir_name = option_dict[NWP_MODEL_TO_DIR_KEY]
    nwp_model_to_field_names = option_dict[NWP_MODEL_TO_FIELDS_KEY]

    first_nwp_model_names = list(nwp_model_to_dir_name.keys())
    second_nwp_model_names = list(nwp_model_to_field_names.keys())
    assert set(first_nwp_model_names) == set(second_nwp_model_names)

    nwp_model_names = list(set(second_nwp_model_names))

    for this_model_name in nwp_model_names:
        error_checking.assert_is_string(nwp_model_to_dir_name[this_model_name])
        error_checking.assert_is_string_list(
            nwp_model_to_field_names[this_model_name]
        )

        for this_field_name in nwp_model_to_field_names[this_model_name]:
            nwp_model_utils.check_field_name(this_field_name)

    if option_dict[NWP_NORM_FILE_KEY] is not None:
        error_checking.assert_file_exists(option_dict[NWP_NORM_FILE_KEY])

    error_checking.assert_is_boolean(option_dict[NWP_USE_QUANTILE_NORM_KEY])

    error_checking.assert_is_integer(option_dict[TARGET_LEAD_TIME_KEY])
    error_checking.assert_is_greater(option_dict[TARGET_LEAD_TIME_KEY], 0)
    error_checking.assert_is_string_list(option_dict[TARGET_FIELDS_KEY])
    for this_field_name in option_dict[TARGET_FIELDS_KEY]:
        urma_utils.check_field_name(this_field_name)

    error_checking.assert_is_string(option_dict[TARGET_DIR_KEY])
    if option_dict[TARGET_NORM_FILE_KEY] is not None:
        error_checking.assert_file_exists(option_dict[TARGET_NORM_FILE_KEY])

    error_checking.assert_is_boolean(option_dict[TARGETS_USE_QUANTILE_NORM_KEY])

    if (
            option_dict[NBM_CONSTANT_FILE_KEY] is None
            or len(option_dict[NBM_CONSTANT_FIELDS_KEY]) == 0
    ):
        option_dict[NBM_CONSTANT_FILE_KEY] = None
        option_dict[NBM_CONSTANT_FIELDS_KEY] = []

    if option_dict[NBM_CONSTANT_FILE_KEY] is not None:
        error_checking.assert_file_exists(option_dict[NBM_CONSTANT_FILE_KEY])
    for this_field_name in option_dict[NBM_CONSTANT_FIELDS_KEY]:
        nbm_constant_utils.check_field_name(this_field_name)

    error_checking.assert_is_integer(option_dict[BATCH_SIZE_KEY])
    # error_checking.assert_is_geq(option_dict[BATCH_SIZE_KEY], 8)
    error_checking.assert_is_not_nan(option_dict[SENTINEL_VALUE_KEY])
    error_checking.assert_is_boolean(option_dict[SUBSET_GRID_KEY])

    error_checking.assert_is_boolean(
        option_dict[PREDICT_DEWPOINT_DEPRESSION_KEY]
    )
    error_checking.assert_is_boolean(option_dict[PREDICT_GUST_FACTOR_KEY])

    if option_dict[PREDICT_DEWPOINT_DEPRESSION_KEY]:
        assert option_dict[TARGET_NORM_FILE_KEY] is None

        assert (
            urma_utils.DEWPOINT_2METRE_NAME in option_dict[TARGET_FIELDS_KEY]
        )
        option_dict[TARGET_FIELDS_KEY].remove(urma_utils.DEWPOINT_2METRE_NAME)
        option_dict[TARGET_FIELDS_KEY].append(urma_utils.DEWPOINT_2METRE_NAME)

    if option_dict[PREDICT_GUST_FACTOR_KEY]:
        assert option_dict[TARGET_NORM_FILE_KEY] is None

        assert (
            urma_utils.WIND_GUST_10METRE_NAME in option_dict[TARGET_FIELDS_KEY]
        )
        option_dict[TARGET_FIELDS_KEY].remove(urma_utils.WIND_GUST_10METRE_NAME)
        option_dict[TARGET_FIELDS_KEY].append(urma_utils.WIND_GUST_10METRE_NAME)

    error_checking.assert_is_boolean(option_dict[DO_RESIDUAL_PREDICTION_KEY])
    if not option_dict[DO_RESIDUAL_PREDICTION_KEY]:
        assert option_dict[TARGET_NORM_FILE_KEY] is None

        option_dict[RESID_BASELINE_MODEL_KEY] = None
        option_dict[RESID_BASELINE_MODEL_DIR_KEY] = None
        option_dict[RESID_BASELINE_LEAD_TIME_KEY] = -1
        return option_dict

    error_checking.assert_is_string(option_dict[RESID_BASELINE_MODEL_KEY])
    error_checking.assert_is_string(option_dict[RESID_BASELINE_MODEL_DIR_KEY])
    error_checking.assert_is_integer(option_dict[RESID_BASELINE_LEAD_TIME_KEY])
    error_checking.assert_is_greater(
        option_dict[RESID_BASELINE_LEAD_TIME_KEY], 0
    )

    resid_baseline_ds_factor = nwp_model_utils.model_to_nbm_downsampling_factor(
        option_dict[RESID_BASELINE_MODEL_KEY]
    )
    error_checking.assert_equals(resid_baseline_ds_factor, 1)

    return option_dict


def _init_predictor_matrices_1example(
        nwp_model_names, nwp_model_to_field_names, num_nwp_lead_times,
        subset_grid):
    """Initializes predictor matrices for one example.

    :param nwp_model_names: 1-D list with names of NWP models.
    :param nwp_model_to_field_names: Dictionary.  For details, see documentation
        for `data_generator`.
    :param num_nwp_lead_times: Number of lead times.
    :param subset_grid: Boolean flag.  If True, will subset grid to smaller
        domain.
    :return: predictor_matrices_2pt5km: 1-D list of numpy arrays for 2.5-km
        resolution.  One array per 2.5-km model.  If there are no 2.5-km models,
        this is None instead of a list.
    :return: predictor_matrices_10km: Same but for 10-km models.
    :return: predictor_matrices_20km: Same but for 20-km models.
    :return: predictor_matrices_40km: Same but for 40-km models.
    """

    downsampling_factors = numpy.array([
        nwp_model_utils.model_to_nbm_downsampling_factor(m)
        for m in nwp_model_names
    ], dtype=int)

    model_indices = numpy.where(downsampling_factors == 1)[0]
    num_rows, num_columns = nwp_model_utils.model_to_nbm_grid_size(
        nwp_model_utils.HRRR_MODEL_NAME
    )

    if subset_grid:
        num_rows = 449
        num_columns = 449

    if len(model_indices) == 0:
        predictor_matrices_2pt5km = None
    else:
        first_dim = (num_rows, num_columns, num_nwp_lead_times)

        predictor_matrices_2pt5km = [
            numpy.full(
                first_dim + (len(nwp_model_to_field_names[nwp_model_names[k]]),),
                numpy.nan
            )
            for k in model_indices
        ]

    model_indices = numpy.where(downsampling_factors == 4)[0]
    num_rows, num_columns = nwp_model_utils.model_to_nbm_grid_size(
        nwp_model_utils.RAP_MODEL_NAME
    )

    if subset_grid:
        num_rows = 113
        num_columns = 113

    if len(model_indices) == 0:
        predictor_matrices_10km = None
    else:
        first_dim = (num_rows, num_columns, num_nwp_lead_times)

        predictor_matrices_10km = [
            numpy.full(
                first_dim + (len(nwp_model_to_field_names[nwp_model_names[k]]),),
                numpy.nan
            )
            for k in model_indices
        ]

    model_indices = numpy.where(downsampling_factors == 8)[0]
    num_rows, num_columns = nwp_model_utils.model_to_nbm_grid_size(
        nwp_model_utils.GFS_MODEL_NAME
    )

    if subset_grid:
        num_rows = 57
        num_columns = 57

    if len(model_indices) == 0:
        predictor_matrices_20km = None
    else:
        first_dim = (num_rows, num_columns, num_nwp_lead_times)

        predictor_matrices_20km = [
            numpy.full(
                first_dim + (len(nwp_model_to_field_names[nwp_model_names[k]]),),
                numpy.nan
            )
            for k in model_indices
        ]

    model_indices = numpy.where(downsampling_factors == 16)[0]
    num_rows, num_columns = nwp_model_utils.model_to_nbm_grid_size(
        nwp_model_utils.GEFS_MODEL_NAME
    )

    if subset_grid:
        num_rows = 29
        num_columns = 29

    if len(model_indices) == 0:
        predictor_matrices_40km = None
    else:
        first_dim = (num_rows, num_columns, num_nwp_lead_times)

        predictor_matrices_40km = [
            numpy.full(
                first_dim + (len(nwp_model_to_field_names[nwp_model_names[k]]),),
                numpy.nan
            )
            for k in model_indices
        ]

    return (
        predictor_matrices_2pt5km, predictor_matrices_10km,
        predictor_matrices_20km, predictor_matrices_40km
    )


def _init_matrices_1batch(
        nwp_model_names, nwp_model_to_field_names, num_nwp_lead_times,
        num_target_fields, num_examples_per_batch, subset_grid,
        do_residual_prediction):
    """Initializes predictor and target matrices for one batch.

    :param nwp_model_names: 1-D list with names of NWP models.
    :param nwp_model_to_field_names: Dictionary.  For details, see documentation
        for `data_generator`.
    :param num_nwp_lead_times: Number of lead times.
    :param num_target_fields: Number of target fields.
    :param num_examples_per_batch: Batch size.
    :param subset_grid: Boolean flag.  If True, will subset grid to smaller
        domain.
    :param do_residual_prediction: Boolean flag.  If True, the NN is predicting
        difference between a given NWP forecast and the URMA truth.  If True,
        the NN is predicting the URMA truth directly.
    :return: predictor_matrix_2pt5km: numpy array for NWP data with 2.5-km
        resolution.  If there are no 2.5-km models, this is None instead of an
        array.
    :return: predictor_matrix_10km: Same but for 10-km models.
    :return: predictor_matrix_20km: Same but for 20-km models.
    :return: predictor_matrix_40km: Same but for 40-km models.
    :return: predictor_matrix_resid_baseline: Same but for residual baseline.
    :return: target_matrix: Same but for target fields.
    """

    downsampling_factors = numpy.array([
        nwp_model_utils.model_to_nbm_downsampling_factor(m)
        for m in nwp_model_names
    ], dtype=int)

    model_indices = numpy.where(downsampling_factors == 1)[0]
    num_rows, num_columns = nwp_model_utils.model_to_nbm_grid_size(
        nwp_model_utils.HRRR_MODEL_NAME
    )

    if subset_grid:
        num_rows = 449
        num_columns = 449

    target_matrix = numpy.full(
        (num_examples_per_batch, num_rows, num_columns, num_target_fields),
        numpy.nan
    )

    if do_residual_prediction:
        predictor_matrix_resid_baseline = numpy.full(
            (num_examples_per_batch, num_rows, num_columns, num_target_fields),
            numpy.nan
        )
    else:
        predictor_matrix_resid_baseline = None

    if len(model_indices) == 0:
        predictor_matrix_2pt5km = None
    else:
        first_dim = (
            num_examples_per_batch, num_rows, num_columns, num_nwp_lead_times
        )

        predictor_matrix_2pt5km = numpy.concatenate([
            numpy.full(
                first_dim + (len(nwp_model_to_field_names[nwp_model_names[k]]),),
                numpy.nan
            )
            for k in model_indices
        ], axis=-1)

    model_indices = numpy.where(downsampling_factors == 4)[0]
    num_rows, num_columns = nwp_model_utils.model_to_nbm_grid_size(
        nwp_model_utils.RAP_MODEL_NAME
    )

    if subset_grid:
        num_rows = 113
        num_columns = 113

    if len(model_indices) == 0:
        predictor_matrix_10km = None
    else:
        first_dim = (
            num_examples_per_batch, num_rows, num_columns, num_nwp_lead_times
        )

        predictor_matrix_10km = numpy.concatenate([
            numpy.full(
                first_dim + (len(nwp_model_to_field_names[nwp_model_names[k]]),),
                numpy.nan
            )
            for k in model_indices
        ], axis=-1)

    model_indices = numpy.where(downsampling_factors == 8)[0]
    num_rows, num_columns = nwp_model_utils.model_to_nbm_grid_size(
        nwp_model_utils.GFS_MODEL_NAME
    )

    if subset_grid:
        num_rows = 57
        num_columns = 57

    if len(model_indices) == 0:
        predictor_matrix_20km = None
    else:
        first_dim = (
            num_examples_per_batch, num_rows, num_columns, num_nwp_lead_times
        )

        predictor_matrix_20km = numpy.concatenate([
            numpy.full(
                first_dim + (len(nwp_model_to_field_names[nwp_model_names[k]]),),
                numpy.nan
            )
            for k in model_indices
        ], axis=-1)

    model_indices = numpy.where(downsampling_factors == 16)[0]
    num_rows, num_columns = nwp_model_utils.model_to_nbm_grid_size(
        nwp_model_utils.GEFS_MODEL_NAME
    )

    if subset_grid:
        num_rows = 29
        num_columns = 29

    if len(model_indices) == 0:
        predictor_matrix_40km = None
    else:
        first_dim = (
            num_examples_per_batch, num_rows, num_columns, num_nwp_lead_times
        )

        predictor_matrix_40km = numpy.concatenate([
            numpy.full(
                first_dim + (len(nwp_model_to_field_names[nwp_model_names[k]]),),
                numpy.nan
            )
            for k in model_indices
        ], axis=-1)

    return (
        predictor_matrix_2pt5km, predictor_matrix_10km,
        predictor_matrix_20km, predictor_matrix_40km,
        predictor_matrix_resid_baseline, target_matrix
    )


def _read_targets_one_example(
        init_time_unix_sec, target_lead_time_hours,
        target_field_names, target_dir_name,
        target_norm_param_table_xarray, use_quantile_norm, subset_grid):
    """Reads target fields for one example.

    NBM = National Blend of Models

    M = number of rows in NBM grid (2.5-km target grid)
    N = number of columns in NBM grid (2.5-km target grid)
    F = number of target fields

    :param init_time_unix_sec: Forecast-initialization time.
    :param target_lead_time_hours: See documentation for `data_generator`.
    :param target_field_names: Same.
    :param target_dir_name: Same.
    :param target_norm_param_table_xarray: xarray table with normalization
        parameters for target variables.  If you do not want to normalize (or
        if the input directory already contains normalized data), this should be
        None.
    :param use_quantile_norm: See documentation for `data_generator`.
    :param subset_grid: Boolean flag.  If True, will subset grid to smaller
        domain.
    :return: target_matrix: M-by-N-by-F numpy array of target values.
    """

    target_valid_time_unix_sec = (
        init_time_unix_sec + HOURS_TO_SECONDS * target_lead_time_hours
    )
    target_valid_date_string = time_conversion.unix_sec_to_string(
        target_valid_time_unix_sec, urma_io.DATE_FORMAT
    )

    urma_file_name = urma_io.find_file(
        directory_name=target_dir_name,
        valid_date_string=target_valid_date_string,
        raise_error_if_missing=True
    )

    print('Reading data from: "{0:s}"...'.format(urma_file_name))
    urma_table_xarray = urma_io.read_file(urma_file_name)
    urma_table_xarray = urma_utils.subset_by_time(
        urma_table_xarray=urma_table_xarray,
        desired_times_unix_sec=
        numpy.array([target_valid_time_unix_sec], dtype=int)
    )
    urma_table_xarray = urma_utils.subset_by_field(
        urma_table_xarray=urma_table_xarray,
        desired_field_names=target_field_names
    )

    if target_norm_param_table_xarray is None:
        data_matrix = urma_table_xarray[urma_utils.DATA_KEY].values

        if urma_utils.TEMPERATURE_2METRE_NAME in target_field_names:
            print('Converting target temperatures from K to deg C...')

            k = numpy.where(
                urma_table_xarray.coords[urma_utils.FIELD_DIM].values ==
                urma_utils.TEMPERATURE_2METRE_NAME
            )[0][0]

            data_matrix[..., k] = temperature_conv.kelvins_to_celsius(
                data_matrix[..., k]
            )

        if urma_utils.DEWPOINT_2METRE_NAME in target_field_names:
            print('Converting target dewpoints from K to deg C...')

            k = numpy.where(
                urma_table_xarray.coords[urma_utils.FIELD_DIM].values ==
                urma_utils.DEWPOINT_2METRE_NAME
            )[0][0]

            data_matrix[..., k] = temperature_conv.kelvins_to_celsius(
                data_matrix[..., k]
            )

        urma_table_xarray = urma_table_xarray.assign({
            urma_utils.DATA_KEY: (
                urma_table_xarray[urma_utils.DATA_KEY].dims, data_matrix
            )
        })
    else:
        print('Normalizing target variables to z-scores...')
        urma_table_xarray = normalization.normalize_targets(
            urma_table_xarray=urma_table_xarray,
            norm_param_table_xarray=target_norm_param_table_xarray,
            use_quantile_norm=use_quantile_norm
        )

    target_matrix = numpy.transpose(
        urma_table_xarray[urma_utils.DATA_KEY].values[0, ...],
        axes=(1, 0, 2)
    )

    if subset_grid:
        target_matrix = target_matrix[544:993, 752:1201]

    return target_matrix


def _read_residual_baseline_one_example(
        init_time_unix_sec, nwp_model_name, nwp_lead_time_hours,
        nwp_directory_name, target_field_names, subset_grid,
        predict_dewpoint_depression, predict_gust_factor):
    """Reads residual baseline for one example.

    M = number of rows in NBM grid (2.5-km target grid)
    N = number of columns in NBM grid (2.5-km target grid)
    F = number of target fields

    Should be used only if the NN is doing residual prediction, i.e., predicting
    the departure between the URMA truth and an NWP forecast.

    :param init_time_unix_sec: Forecast-initialization time.
    :param nwp_model_name: Name of NWP model used to create residual baseline.
    :param nwp_lead_time_hours: NWP lead time used to create residual baseline.
    :param nwp_directory_name: Path to NWP data.  Relevant files therein will be
        found by `interp_nwp_model_io.find_file`.
    :param target_field_names: See documentation for `data_generator`.
    :param subset_grid: Boolean flag.  If True, will subset grid to smaller
        domain.
    :param predict_dewpoint_depression: Boolean flag.  If True, the NN is
        predicting dewpoint depression instead of dewpoint.
    :param predict_gust_factor: Boolean flag.  If True, the NN is predicting
        gust factor instead of gust speed.
    :return: residual_baseline_matrix: M-by-N-by-F numpy array of baseline
        predictions.  These are all unnormalized.
    """

    target_field_to_baseline_nwp_field = {
        urma_utils.TEMPERATURE_2METRE_NAME:
            nwp_model_utils.TEMPERATURE_2METRE_NAME,
        urma_utils.DEWPOINT_2METRE_NAME: nwp_model_utils.DEWPOINT_2METRE_NAME,
        urma_utils.U_WIND_10METRE_NAME: nwp_model_utils.U_WIND_10METRE_NAME,
        urma_utils.V_WIND_10METRE_NAME: nwp_model_utils.V_WIND_10METRE_NAME,
        urma_utils.WIND_GUST_10METRE_NAME: nwp_model_utils.V_WIND_10METRE_NAME
    }

    if nwp_model_name == nwp_model_utils.GRIDDED_LAMP_MODEL_NAME:
        target_field_to_baseline_nwp_field[
            urma_utils.WIND_GUST_10METRE_NAME
        ] = nwp_model_utils.WIND_GUST_10METRE_NAME

    input_file_name = interp_nwp_model_io.find_file(
        directory_name=nwp_directory_name,
        init_time_unix_sec=init_time_unix_sec,
        forecast_hour=nwp_lead_time_hours,
        model_name=nwp_model_name,
        raise_error_if_missing=False
    )

    if not os.path.isfile(input_file_name):
        warning_string = (
            'POTENTIAL ERROR: Could not find file expected at: "{0:s}".  This '
            'is needed for residual baseline.'
        ).format(input_file_name)

        warnings.warn(warning_string)
        return None

    print('Reading data from: "{0:s}"...'.format(input_file_name))
    nwp_forecast_table_xarray = interp_nwp_model_io.read_file(input_file_name)

    if subset_grid:
        nwp_forecast_table_xarray = nwp_model_utils.subset_by_row(
            nwp_forecast_table_xarray=nwp_forecast_table_xarray,
            desired_row_indices=numpy.linspace(544, 992, num=449, dtype=int)
        )
        nwp_forecast_table_xarray = nwp_model_utils.subset_by_column(
            nwp_forecast_table_xarray=nwp_forecast_table_xarray,
            desired_column_indices=numpy.linspace(752, 1200, num=449, dtype=int)
        )

    if predict_dewpoint_depression:
        nwp_forecast_table_xarray = __nwp_2m_dewpoint_to_depression(
            nwp_forecast_table_xarray
        )
    if predict_gust_factor:
        nwp_forecast_table_xarray = __nwp_10m_gust_speed_to_factor(
            nwp_forecast_table_xarray
        )

    nwp_forecast_table_xarray = __nwp_2m_temp_to_celsius(
        nwp_forecast_table_xarray
    )

    these_matrices = [
        nwp_model_utils.get_field(
            nwp_forecast_table_xarray=nwp_forecast_table_xarray,
            field_name=target_field_to_baseline_nwp_field[f]
        )[0, ...]
        for f in target_field_names
    ]
    residual_baseline_matrix = numpy.stack(these_matrices, axis=-1)

    need_fake_gust_data = (
        urma_utils.WIND_GUST_10METRE_NAME in target_field_names
        and nwp_model_name != nwp_model_utils.GRIDDED_LAMP_MODEL_NAME
    )

    if not need_fake_gust_data:
        return residual_baseline_matrix

    gust_idx = target_field_names.index(urma_utils.WIND_GUST_10METRE_NAME)

    if predict_gust_factor:
        residual_baseline_matrix[..., gust_idx] = DEFAULT_GUST_FACTOR - 1.
        return residual_baseline_matrix

    u_idx = target_field_names.index(urma_utils.U_WIND_10METRE_NAME)
    v_idx = target_field_names.index(urma_utils.V_WIND_10METRE_NAME)

    speed_matrix = numpy.sqrt(
        residual_baseline_matrix[..., u_idx] ** 2 +
        residual_baseline_matrix[..., v_idx] ** 2
    )
    residual_baseline_matrix[..., gust_idx] = (
        DEFAULT_GUST_FACTOR * speed_matrix
    )

    return residual_baseline_matrix


def _read_predictors_one_example(
        init_time_unix_sec, nwp_model_names, nwp_lead_times_hours,
        nwp_model_to_field_names, nwp_model_to_dir_name,
        nwp_norm_param_table_xarray, use_quantile_norm, subset_grid):
    """Reads predictor fields for one example.

    :param init_time_unix_sec: Forecast-initialization time.
    :param nwp_model_names: 1-D list with names of NWP models used to create
        predictors.
    :param nwp_lead_times_hours: See documentation for `data_generator`.
    :param nwp_model_to_field_names: Same.
    :param nwp_model_to_dir_name: Same.
    :param nwp_norm_param_table_xarray: xarray table with normalization
        parameters for predictor variables.  If you do not want to normalize (or
        if the input directory already contains normalized data), this should be
        None.
    :param use_quantile_norm: See documentation for `data_generator`.
    :param subset_grid: Boolean flag.  If True, will subset grid to smaller
        domain.
    :return: predictor_matrix_2pt5km: Same as output from `data_generator` but
        without first axis.
    :return: predictor_matrix_10km: Same as output from `data_generator` but
        without first axis.
    :return: predictor_matrix_20km: Same as output from `data_generator` but
        without first axis.
    :return: predictor_matrix_40km: Same as output from `data_generator` but
        without first axis.
    """

    num_nwp_models = len(nwp_model_names)
    num_nwp_lead_times = len(nwp_lead_times_hours)

    nwp_downsampling_factors = numpy.array([
        nwp_model_utils.model_to_nbm_downsampling_factor(m)
        for m in nwp_model_names
    ], dtype=int)

    (
        predictor_matrices_2pt5km, predictor_matrices_10km,
        predictor_matrices_20km, predictor_matrices_40km
    ) = _init_predictor_matrices_1example(
        nwp_model_names=nwp_model_names,
        nwp_model_to_field_names=nwp_model_to_field_names,
        num_nwp_lead_times=num_nwp_lead_times,
        subset_grid=subset_grid
    )

    for i in range(num_nwp_models):
        for j in range(num_nwp_lead_times):
            this_file_name = interp_nwp_model_io.find_file(
                directory_name=nwp_model_to_dir_name[nwp_model_names[i]],
                init_time_unix_sec=init_time_unix_sec,
                forecast_hour=nwp_lead_times_hours[j],
                model_name=nwp_model_names[i],
                raise_error_if_missing=False
            )

            # TODO(thunderhoser): For now, letting all missing files go.  Not
            # sure if this is always the best decision, though.
            if not os.path.isfile(this_file_name):
                warning_string = (
                    'POTENTIAL ERROR: Could not find file expected at: "{0:s}".'
                    '  Filling predictor matrix with NaN, instead.'
                ).format(this_file_name)

                warnings.warn(warning_string)
                continue

            print('Reading data from: "{0:s}"...'.format(this_file_name))
            nwp_forecast_table_xarray = interp_nwp_model_io.read_file(
                this_file_name
            )
            nwp_forecast_table_xarray = nwp_model_utils.subset_by_field(
                nwp_forecast_table_xarray=nwp_forecast_table_xarray,
                desired_field_names=nwp_model_to_field_names[nwp_model_names[i]]
            )

            if subset_grid:
                if nwp_downsampling_factors[i] == 1:
                    nwp_forecast_table_xarray = nwp_model_utils.subset_by_row(
                        nwp_forecast_table_xarray=nwp_forecast_table_xarray,
                        desired_row_indices=
                        numpy.linspace(544, 992, num=449, dtype=int)
                    )
                    nwp_forecast_table_xarray = nwp_model_utils.subset_by_column(
                        nwp_forecast_table_xarray=nwp_forecast_table_xarray,
                        desired_column_indices=
                        numpy.linspace(752, 1200, num=449, dtype=int)
                    )
                elif nwp_downsampling_factors[i] == 4:
                    nwp_forecast_table_xarray = nwp_model_utils.subset_by_row(
                        nwp_forecast_table_xarray=nwp_forecast_table_xarray,
                        desired_row_indices=
                        numpy.linspace(136, 248, num=113, dtype=int)
                    )
                    nwp_forecast_table_xarray = nwp_model_utils.subset_by_column(
                        nwp_forecast_table_xarray=nwp_forecast_table_xarray,
                        desired_column_indices=
                        numpy.linspace(188, 300, num=113, dtype=int)
                    )
                elif nwp_downsampling_factors[i] == 8:
                    nwp_forecast_table_xarray = nwp_model_utils.subset_by_row(
                        nwp_forecast_table_xarray=nwp_forecast_table_xarray,
                        desired_row_indices=
                        numpy.linspace(68, 124, num=57, dtype=int)
                    )
                    nwp_forecast_table_xarray = nwp_model_utils.subset_by_column(
                        nwp_forecast_table_xarray=nwp_forecast_table_xarray,
                        desired_column_indices=
                        numpy.linspace(94, 150, num=57, dtype=int)
                    )
                else:
                    nwp_forecast_table_xarray = nwp_model_utils.subset_by_row(
                        nwp_forecast_table_xarray=nwp_forecast_table_xarray,
                        desired_row_indices=
                        numpy.linspace(34, 62, num=29, dtype=int)
                    )
                    nwp_forecast_table_xarray = nwp_model_utils.subset_by_column(
                        nwp_forecast_table_xarray=nwp_forecast_table_xarray,
                        desired_column_indices=
                        numpy.linspace(47, 75, num=29, dtype=int)
                    )

            if nwp_norm_param_table_xarray is not None:
                print('Normalizing predictor variables to z-scores...')
                nwp_forecast_table_xarray = (
                    normalization.normalize_nwp_data(
                        nwp_forecast_table_xarray=nwp_forecast_table_xarray,
                        norm_param_table_xarray=nwp_norm_param_table_xarray,
                        use_quantile_norm=use_quantile_norm
                    )
                )

            nwpft = nwp_forecast_table_xarray
            matrix_index = numpy.sum(
                nwp_downsampling_factors[:i] == nwp_downsampling_factors[i]
            )

            if nwp_downsampling_factors[i] == 1:
                predictor_matrices_2pt5km[matrix_index][..., j, :] = (
                    nwpft[nwp_model_utils.DATA_KEY].values[0, ...]
                )
            elif nwp_downsampling_factors[i] == 4:
                predictor_matrices_10km[matrix_index][..., j, :] = (
                    nwpft[nwp_model_utils.DATA_KEY].values[0, ...]
                )
            elif nwp_downsampling_factors[i] == 8:
                predictor_matrices_20km[matrix_index][..., j, :] = (
                    nwpft[nwp_model_utils.DATA_KEY].values[0, ...]
                )
            else:
                predictor_matrices_40km[matrix_index][..., j, :] = (
                    nwpft[nwp_model_utils.DATA_KEY].values[0, ...]
                )

    if predictor_matrices_2pt5km is None:
        predictor_matrix_2pt5km = None
    else:
        predictor_matrix_2pt5km = numpy.concatenate(
            predictor_matrices_2pt5km, axis=-1
        )

    if predictor_matrices_10km is None:
        predictor_matrix_10km = None
    else:
        predictor_matrix_10km = numpy.concatenate(
            predictor_matrices_10km, axis=-1
        )

    if predictor_matrices_20km is None:
        predictor_matrix_20km = None
    else:
        predictor_matrix_20km = numpy.concatenate(
            predictor_matrices_20km, axis=-1
        )

    if predictor_matrices_40km is None:
        predictor_matrix_40km = None
    else:
        predictor_matrix_40km = numpy.concatenate(
            predictor_matrices_40km, axis=-1
        )

    return (
        predictor_matrix_2pt5km, predictor_matrix_10km,
        predictor_matrix_20km, predictor_matrix_40km
    )


def create_data(option_dict):
    """Creates, instead of generates, neural-network inputs.

    E = number of examples returned

    :param option_dict: See documentation for `data_generator`.
    :return: data_dict: Dictionary with the following keys.
    data_dict["predictor_matrices"]: Same as output from `data_generator`.
    data_dict["target_matrix"]: Same as output from `data_generator`.
    data_dict["init_times_unix_sec"]: length-E numpy array of forecast-
        initialization times.
    """

    # Check input args.
    option_dict[BATCH_SIZE_KEY] = 32  # Dummy argument.

    option_dict = _check_generator_args(option_dict)
    first_init_times_unix_sec = option_dict[FIRST_INIT_TIMES_KEY]
    last_init_times_unix_sec = option_dict[LAST_INIT_TIMES_KEY]
    nwp_lead_times_hours = option_dict[NWP_LEAD_TIMES_KEY]
    nwp_model_to_dir_name = option_dict[NWP_MODEL_TO_DIR_KEY]
    nwp_model_to_field_names = option_dict[NWP_MODEL_TO_FIELDS_KEY]
    nwp_normalization_file_name = option_dict[NWP_NORM_FILE_KEY]
    nwp_use_quantile_norm = option_dict[NWP_USE_QUANTILE_NORM_KEY]
    target_lead_time_hours = option_dict[TARGET_LEAD_TIME_KEY]
    target_field_names = option_dict[TARGET_FIELDS_KEY]
    target_dir_name = option_dict[TARGET_DIR_KEY]
    target_normalization_file_name = option_dict[TARGET_NORM_FILE_KEY]
    targets_use_quantile_norm = option_dict[TARGETS_USE_QUANTILE_NORM_KEY]
    nbm_constant_field_names = option_dict[NBM_CONSTANT_FIELDS_KEY]
    nbm_constant_file_name = option_dict[NBM_CONSTANT_FILE_KEY]
    sentinel_value = option_dict[SENTINEL_VALUE_KEY]
    subset_grid = option_dict[SUBSET_GRID_KEY]

    do_residual_prediction = option_dict[DO_RESIDUAL_PREDICTION_KEY]
    predict_dewpoint_depression = option_dict[PREDICT_DEWPOINT_DEPRESSION_KEY]
    predict_gust_factor = option_dict[PREDICT_GUST_FACTOR_KEY]
    resid_baseline_model_name = option_dict[RESID_BASELINE_MODEL_KEY]
    resid_baseline_model_dir_name = option_dict[RESID_BASELINE_MODEL_DIR_KEY]
    resid_baseline_lead_time_hours = option_dict[RESID_BASELINE_LEAD_TIME_KEY]

    first_nwp_model_names = list(nwp_model_to_dir_name.keys())
    second_nwp_model_names = list(nwp_model_to_field_names.keys())
    assert set(first_nwp_model_names) == set(second_nwp_model_names)

    nwp_model_names = list(set(first_nwp_model_names))
    nwp_model_names = [
        m for m in nwp_model_names if m != nwp_model_utils.WRF_ARW_MODEL_NAME
    ]

    if nwp_normalization_file_name is None:
        nwp_norm_param_table_xarray = None
    else:
        print('Reading normalization params from: "{0:s}"...'.format(
            nwp_normalization_file_name
        ))
        nwp_norm_param_table_xarray = nwp_model_io.read_normalization_file(
            nwp_normalization_file_name
        )

    if target_normalization_file_name is None:
        target_norm_param_table_xarray = None
    else:
        print('Reading normalization params from: "{0:s}"...'.format(
            target_normalization_file_name
        ))
        target_norm_param_table_xarray = urma_io.read_normalization_file(
            target_normalization_file_name
        )

    init_time_intervals_sec = numpy.array([
        nwp_model_utils.model_to_init_time_interval(m) for m in nwp_model_names
    ], dtype=int)

    init_times_unix_sec = numpy.concatenate([
        time_periods.range_and_interval_to_list(
            start_time_unix_sec=f,
            end_time_unix_sec=l,
            time_interval_sec=numpy.max(init_time_intervals_sec),
            include_endpoint=True
        )
        for f, l in zip(first_init_times_unix_sec, last_init_times_unix_sec)
    ])

    # Do actual stuff.
    num_examples = len(init_times_unix_sec)

    # Do actual stuff.
    if nbm_constant_file_name is None:
        nbm_constant_matrix = None
    else:
        print('Reading data from: "{0:s}"...'.format(nbm_constant_file_name))
        nbm_constant_table_xarray = nbm_constant_io.read_file(
            nbm_constant_file_name
        )
        nbmct = nbm_constant_table_xarray

        field_indices = numpy.array([
            numpy.where(
                nbmct.coords[nbm_constant_utils.FIELD_DIM].values == f
            )[0][0]
            for f in nbm_constant_field_names
        ], dtype=int)

        nbm_constant_matrix = (
            nbmct[nbm_constant_utils.DATA_KEY].values[..., field_indices]
        )
        if subset_grid:
            nbm_constant_matrix = nbm_constant_matrix[544:993, 752:1201]

    good_example_flags = numpy.full(num_examples, True, dtype=bool)

    (
        predictor_matrix_2pt5km, predictor_matrix_10km,
        predictor_matrix_20km, predictor_matrix_40km,
        predictor_matrix_resid_baseline, target_matrix
    ) = _init_matrices_1batch(
        nwp_model_names=nwp_model_names,
        nwp_model_to_field_names=nwp_model_to_field_names,
        num_nwp_lead_times=len(nwp_lead_times_hours),
        num_target_fields=len(target_field_names),
        num_examples_per_batch=num_examples,
        subset_grid=subset_grid,
        do_residual_prediction=do_residual_prediction
    )

    for i in range(num_examples):
        this_target_matrix = _read_targets_one_example(
            init_time_unix_sec=init_times_unix_sec[i],
            target_lead_time_hours=target_lead_time_hours,
            target_field_names=target_field_names,
            target_dir_name=target_dir_name,
            target_norm_param_table_xarray=target_norm_param_table_xarray,
            use_quantile_norm=targets_use_quantile_norm,
            subset_grid=subset_grid
        )

        if this_target_matrix is None:
            good_example_flags[i] = False
            continue

        target_matrix[i, ...] = this_target_matrix

        if do_residual_prediction:
            this_baseline_matrix = _read_residual_baseline_one_example(
                init_time_unix_sec=init_times_unix_sec[i],
                nwp_model_name=resid_baseline_model_name,
                nwp_lead_time_hours=resid_baseline_lead_time_hours,
                nwp_directory_name=resid_baseline_model_dir_name,
                target_field_names=target_field_names,
                subset_grid=subset_grid,
                predict_dewpoint_depression=predict_dewpoint_depression,
                predict_gust_factor=predict_gust_factor
            )

            if this_baseline_matrix is None:
                good_example_flags[i] = False
                continue

            predictor_matrix_resid_baseline[i, ...] = this_baseline_matrix

        (
            this_predictor_matrix_2pt5km,
            this_predictor_matrix_10km,
            this_predictor_matrix_20km,
            this_predictor_matrix_40km
        ) = _read_predictors_one_example(
            init_time_unix_sec=init_times_unix_sec[i],
            nwp_model_names=nwp_model_names,
            nwp_lead_times_hours=nwp_lead_times_hours,
            nwp_model_to_field_names=nwp_model_to_field_names,
            nwp_model_to_dir_name=nwp_model_to_dir_name,
            nwp_norm_param_table_xarray=nwp_norm_param_table_xarray,
            use_quantile_norm=nwp_use_quantile_norm,
            subset_grid=subset_grid
        )

        if predictor_matrix_2pt5km is not None:
            good_example_flags[i] &= not numpy.all(
                numpy.isnan(this_predictor_matrix_2pt5km)
            )
            predictor_matrix_2pt5km[i, ...] = this_predictor_matrix_2pt5km
        if predictor_matrix_10km is not None:
            good_example_flags[i] &= not numpy.all(
                numpy.isnan(this_predictor_matrix_10km)
            )
            predictor_matrix_10km[i, ...] = this_predictor_matrix_10km
        if predictor_matrix_20km is not None:
            good_example_flags[i] &= not numpy.all(
                numpy.isnan(this_predictor_matrix_20km)
            )
            predictor_matrix_20km[i, ...] = this_predictor_matrix_20km
        if predictor_matrix_40km is not None:
            good_example_flags[i] &= not numpy.all(
                numpy.isnan(this_predictor_matrix_40km)
            )
            predictor_matrix_40km[i, ...] = this_predictor_matrix_40km

    good_indices = numpy.where(good_example_flags)[0]
    init_times_unix_sec = init_times_unix_sec[good_indices, ...]
    target_matrix = target_matrix[good_indices, ...]
    error_checking.assert_is_numpy_array_without_nan(target_matrix)

    print((
        'Shape of 2.5-km target matrix and NaN fraction: '
        '{0:s}, {1:.04f}'
    ).format(
        str(target_matrix.shape),
        numpy.mean(numpy.isnan(target_matrix))
    ))

    if predictor_matrix_2pt5km is not None:
        predictor_matrix_2pt5km = predictor_matrix_2pt5km[good_indices, ...]

        print((
            'Shape of 2.5-km predictor matrix and NaN fraction: '
            '{0:s}, {1:.04f}'
        ).format(
            str(predictor_matrix_2pt5km.shape),
            numpy.mean(numpy.isnan(predictor_matrix_2pt5km))
        ))

        predictor_matrix_2pt5km[numpy.isnan(predictor_matrix_2pt5km)] = (
            sentinel_value
        )

    if predictor_matrix_resid_baseline is not None:
        predictor_matrix_resid_baseline = predictor_matrix_resid_baseline[
            good_indices, ...
        ]

        print((
            'Shape of residual baseline matrix and NaN fraction: '
            '{0:s}, {1:.04f}'
        ).format(
            str(predictor_matrix_resid_baseline.shape),
            numpy.mean(numpy.isnan(predictor_matrix_resid_baseline))
        ))

        error_checking.assert_is_numpy_array_without_nan(
            predictor_matrix_resid_baseline
        )

    if nbm_constant_matrix is not None:
        nbm_constant_matrix = numpy.repeat(
            numpy.expand_dims(nbm_constant_matrix, axis=0),
            axis=0, repeats=len(good_indices)
        )
        print('Shape of NBM-constant matrix: {0:s}'.format(
            str(nbm_constant_matrix.shape)
        ))

    if predictor_matrix_10km is not None:
        predictor_matrix_10km = predictor_matrix_10km[good_indices, ...]

        print((
            'Shape of 10-km predictor matrix and NaN fraction: '
            '{0:s}, {1:.04f}'
        ).format(
            str(predictor_matrix_10km.shape),
            numpy.mean(numpy.isnan(predictor_matrix_10km))
        ))

        predictor_matrix_10km[numpy.isnan(predictor_matrix_10km)] = (
            sentinel_value
        )

    if predictor_matrix_20km is not None:
        predictor_matrix_20km = predictor_matrix_20km[good_indices, ...]

        print((
            'Shape of 20-km predictor matrix and NaN fraction: '
            '{0:s}, {1:.04f}'
        ).format(
            str(predictor_matrix_20km.shape),
            numpy.mean(numpy.isnan(predictor_matrix_20km))
        ))

        predictor_matrix_20km[numpy.isnan(predictor_matrix_20km)] = (
            sentinel_value
        )

    if predictor_matrix_40km is not None:
        predictor_matrix_40km = predictor_matrix_40km[good_indices, ...]

        print((
            'Shape of 40-km predictor matrix and NaN fraction: '
            '{0:s}, {1:.04f}'
        ).format(
            str(predictor_matrix_40km.shape),
            numpy.mean(numpy.isnan(predictor_matrix_40km))
        ))

        predictor_matrix_40km[numpy.isnan(predictor_matrix_40km)] = (
            sentinel_value
        )

    predictor_matrices = [
        m for m in [
            predictor_matrix_2pt5km, nbm_constant_matrix,
            predictor_matrix_10km, predictor_matrix_20km,
            predictor_matrix_40km, predictor_matrix_resid_baseline
        ]
        if m is not None
    ]

    predictor_matrices = [p.astype('float32') for p in predictor_matrices]

    return {
        PREDICTOR_MATRICES_KEY: predictor_matrices,
        TARGET_MATRIX_KEY: target_matrix,
        INIT_TIMES_KEY: init_times_unix_sec
    }


def data_generator(option_dict):
    """Generates training or validation data for neural network.

    Generators should be used only at training time, not at inference time.

    NBM = National Blend of Models

    E = number of examples per batch = "batch size"
    M = number of rows in NBM grid (2.5-km target grid)
    N = number of columns in NBM grid (2.5-km target grid)
    P = number of NWP fields (predictor variables) at 2.5-km resolution
    C = number of constant fields (at 2.5-km resolution)
    L = number of NWP lead times
    F = number of target fields

    m = number of rows in 10-km grid
    n = number of columns in 10-km grid
    p = number of NWP fields at 10-km resolution
    mm = number of rows in 20-km grid
    nn = number of columns in 20-km grid
    pp = number of NWP fields at 20-km resolution
    mmm = number of rows in 40-km grid
    nnn = number of columns in 40-km grid
    ppp = number of NWP fields at 40-km resolution

    :param option_dict: Dictionary with the following keys.
    option_dict["first_init_times_unix_sec"]: length-P numpy array (where P =
        number of continuous periods in dataset), containing start time of each
        continuous period.
    option_dict["last_init_times_unix_sec"]: length-P numpy array (where P =
        number of continuous periods in dataset), containing end time of each
        continuous period.
    option_dict["nwp_lead_times_hours"]: 1-D numpy array of lead times for
        NWP-based predictors.
    option_dict["nwp_model_to_dir_name"]: Dictionary, where each key is the name
        of an NWP model used to create predictors and the corresponding value is
        the directory path for data from said model.  NWP-model names must be
        accepted by `nwp_model_utils.check_model_name`, and within each
        directory, relevant files will be found by
        `interp_nwp_model_io.find_file`.
    option_dict["nwp_model_to_field_names"]: Dictionary, where each key is
        the name of an NWP model used to create predictors and the corresponding
        value is a 1-D list, containing fields from said model to be used as
        predictors.  NWP-model names must be accepted by
        `nwp_model_utils.check_model_name`, and field names must be accepted by
        `nwp_model_utils.check_field_name`.
    option_dict["nwp_normalization_file_name"]: Path to file with normalization
        params for NWP data (readable by
        `nwp_model_io.read_normalization_file`).  If you do not want to
        normalize NWP predictors, make this None.
    option_dict["nwp_use_quantile_norm"]: Boolean flag.  If True, will normalize
        NWP predictors in two steps: quantiles, then z-scores.  If False, will
        do simple z-score normalization.
    option_dict["target_lead_time_hours"]: Lead time for target fields.
    option_dict["target_field_names"]: length-F list with names of target
        fields.  Each must be accepted by `urma_utils.check_field_name`.
    option_dict["target_dir_name"]: Path to directory with target fields (i.e.,
        URMA data).  Files within this directory will be found by
        `urma_io.find_file` and read by `urma_io.read_file`.
    option_dict["target_normalization_file_name"]: Path to file with
        normalization params for target fields (readable by
        `urma_io.read_normalization_file`).  If you do not want to normalize
        targets, make this None.
    option_dict["targets_use_quantile_norm"]: Same as "nwp_use_quantile_norm"
        but for target fields.
    option_dict["nbm_constant_field_names"]: length-C list with names of NBM
        constant fields, to be used as predictors.  Each must be accepted by
        `nbm_constant_utils.check_field_name`.  If you do not want NBM-constant
        predictors, make this an empty list.
    option_dict["nbm_constant_file_name"]: Path to file with NBM constant
        fields (readable by `nbm_constant_io.read_file`).  If you do not want
        NBM-constant predictors, make this None.
    option_dict["num_examples_per_batch"]: Number of data examples per batch,
        usually just called "batch size".
    option_dict["sentinel_value"]: All NaN will be replaced with this value.
    option_dict["subset_grid"]: Boolean flag.  If True, will subset full grid to
        smaller domain.
    option_dict["predict_dewpoint_depression"]: Boolean flag.  If True, the NN
        is trained to predict dewpoint depression, rather than predicting
        dewpoint temperature directly.
    option_dict["predict_gust_factor"]: Boolean flag.  If True, the NN is
        trained to predict gust factor, rather than predicting gust speed
        directly.
    option_dict["do_residual_prediction"]: Boolean flag.  If True, the NN is
        trained to predict a residual -- i.e., the departure between URMA truth
        and a single NWP forecast.  If False, the NN is trained to predict the
        URMA target fields directly.
    option_dict["resid_baseline_model_name"]: Name of NWP model used to
        generate residual baseline fields.  If do_residual_prediction == False,
        make this argument None.
    option_dict["resid_baseline_lead_time_hours"]: Lead time used to generate
        residual baseline fields.  If do_residual_prediction == False, make this
        argument None.
    option_dict["resid_baseline_model_dir_name"]: Directory path for residual
        baseline fields.  Within this directory, relevant files will be found by
        `interp_nwp_model_io.find_file`.

    :return: predictor_matrices: List with the following items.  Some items may
        be missing.

    predictor_matrices[0]: E-by-M-by-N-by-L-by-P numpy array of predictors at
        2.5-km resolution.
    predictor_matrices[1]: E-by-M-by-N-by-C numpy array of NBM-constant
        predictors, also at 2.5-km resolution.
    predictor_matrices[2]: E-by-m-by-n-by-L-by-p numpy array of predictors at
        10-km resolution.
    predictor_matrices[3]: E-by-mm-by-nn-by-L-by-pp numpy array of predictors at
        20-km resolution.
    predictor_matrices[4]: E-by-mmm-by-nnn-by-L-by-ppp numpy array of predictors
        at 40-km resolution.
    predictor_matrices[5]: E-by-M-by-N-by-F numpy array of baseline values for
        residual prediction.

    :return: target_matrix: E-by-M-by-N-by-F numpy array of targets at 2.5-km
        resolution.
    """

    # TODO(thunderhoser): Also, I should eventually allow a different lead-time
    # set for each NWP model.  To make everything consistent -- i.e., to make
    # sure that feature maps from different NWP models have the same length on
    # the time axis -- I could use either convolution or some kind of
    # interpolation layer(s).  Based on a Google search, it looks like conv is
    # my best option.  Interpolation can be done via Upsampling1D, but the
    # upsampling factor must be an integer.  But also, maybe I could use LSTM to
    # change the sequence length.

    option_dict = _check_generator_args(option_dict)
    first_init_times_unix_sec = option_dict[FIRST_INIT_TIMES_KEY]
    last_init_times_unix_sec = option_dict[LAST_INIT_TIMES_KEY]
    nwp_lead_times_hours = option_dict[NWP_LEAD_TIMES_KEY]
    nwp_model_to_dir_name = option_dict[NWP_MODEL_TO_DIR_KEY]
    nwp_model_to_field_names = option_dict[NWP_MODEL_TO_FIELDS_KEY]
    nwp_normalization_file_name = option_dict[NWP_NORM_FILE_KEY]
    nwp_use_quantile_norm = option_dict[NWP_USE_QUANTILE_NORM_KEY]
    target_lead_time_hours = option_dict[TARGET_LEAD_TIME_KEY]
    target_field_names = option_dict[TARGET_FIELDS_KEY]
    target_dir_name = option_dict[TARGET_DIR_KEY]
    target_normalization_file_name = option_dict[TARGET_NORM_FILE_KEY]
    targets_use_quantile_norm = option_dict[TARGETS_USE_QUANTILE_NORM_KEY]
    nbm_constant_field_names = option_dict[NBM_CONSTANT_FIELDS_KEY]
    nbm_constant_file_name = option_dict[NBM_CONSTANT_FILE_KEY]
    num_examples_per_batch = option_dict[BATCH_SIZE_KEY]
    sentinel_value = option_dict[SENTINEL_VALUE_KEY]
    subset_grid = option_dict[SUBSET_GRID_KEY]

    do_residual_prediction = option_dict[DO_RESIDUAL_PREDICTION_KEY]
    predict_dewpoint_depression = option_dict[PREDICT_DEWPOINT_DEPRESSION_KEY]
    predict_gust_factor = option_dict[PREDICT_GUST_FACTOR_KEY]
    resid_baseline_model_name = option_dict[RESID_BASELINE_MODEL_KEY]
    resid_baseline_model_dir_name = option_dict[RESID_BASELINE_MODEL_DIR_KEY]
    resid_baseline_lead_time_hours = option_dict[RESID_BASELINE_LEAD_TIME_KEY]

    first_nwp_model_names = list(nwp_model_to_dir_name.keys())
    second_nwp_model_names = list(nwp_model_to_field_names.keys())
    assert set(first_nwp_model_names) == set(second_nwp_model_names)

    nwp_model_names = list(set(first_nwp_model_names))
    nwp_model_names = [
        m for m in nwp_model_names if m != nwp_model_utils.WRF_ARW_MODEL_NAME
    ]

    if nwp_normalization_file_name is None:
        nwp_norm_param_table_xarray = None
    else:
        print('Reading normalization params from: "{0:s}"...'.format(
            nwp_normalization_file_name
        ))
        nwp_norm_param_table_xarray = nwp_model_io.read_normalization_file(
            nwp_normalization_file_name
        )

    if target_normalization_file_name is None:
        target_norm_param_table_xarray = None
    else:
        print('Reading normalization params from: "{0:s}"...'.format(
            target_normalization_file_name
        ))
        target_norm_param_table_xarray = urma_io.read_normalization_file(
            target_normalization_file_name
        )

    # TODO(thunderhoser): Different NWP models are available at different init
    # times.  Currently, I handle this by using only common init times.
    # However, I should eventually come up with something more clever.
    init_time_intervals_sec = numpy.array([
        nwp_model_utils.model_to_init_time_interval(m) for m in nwp_model_names
    ], dtype=int)

    init_times_unix_sec = numpy.concatenate([
        time_periods.range_and_interval_to_list(
            start_time_unix_sec=f,
            end_time_unix_sec=l,
            time_interval_sec=numpy.max(init_time_intervals_sec),
            include_endpoint=True
        )
        for f, l in zip(first_init_times_unix_sec, last_init_times_unix_sec)
    ])

    # TODO(thunderhoser): HACK because I have data for only every 5th day right
    # now.

    # TODO(thunderhoser): This will fuck up in leap years!
    init_date_strings = [
        time_conversion.unix_sec_to_string(t, '%Y-%j')
        for t in init_times_unix_sec
    ]
    init_dates_julian = numpy.array(
        [int(t.split('-')[1]) for t in init_date_strings],
        dtype=int
    )
    good_indices = numpy.where(
        numpy.mod(init_dates_julian, 5) == 0
    )[0]

    init_times_unix_sec = init_times_unix_sec[good_indices]
    del init_date_strings
    del init_dates_julian

    # Do actual stuff.
    if nbm_constant_file_name is None:
        nbm_constant_matrix = None
    else:
        print('Reading data from: "{0:s}"...'.format(nbm_constant_file_name))
        nbm_constant_table_xarray = nbm_constant_io.read_file(
            nbm_constant_file_name
        )
        nbmct = nbm_constant_table_xarray

        field_indices = numpy.array([
            numpy.where(
                nbmct.coords[nbm_constant_utils.FIELD_DIM].values == f
            )[0][0]
            for f in nbm_constant_field_names
        ], dtype=int)

        nbm_constant_matrix = (
            nbmct[nbm_constant_utils.DATA_KEY].values[..., field_indices]
        )
        if subset_grid:
            nbm_constant_matrix = nbm_constant_matrix[544:993, 752:1201]

        nbm_constant_matrix = numpy.repeat(
            numpy.expand_dims(nbm_constant_matrix, axis=0),
            axis=0, repeats=num_examples_per_batch
        )

    init_time_index = len(init_times_unix_sec)

    while True:
        (
            predictor_matrix_2pt5km, predictor_matrix_10km,
            predictor_matrix_20km, predictor_matrix_40km,
            predictor_matrix_resid_baseline, target_matrix
        ) = _init_matrices_1batch(
            nwp_model_names=nwp_model_names,
            nwp_model_to_field_names=nwp_model_to_field_names,
            num_nwp_lead_times=len(nwp_lead_times_hours),
            num_target_fields=len(target_field_names),
            num_examples_per_batch=num_examples_per_batch,
            subset_grid=subset_grid,
            do_residual_prediction=do_residual_prediction
        )

        num_examples_in_memory = 0

        while num_examples_in_memory < num_examples_per_batch:
            if init_time_index == len(init_times_unix_sec):
                numpy.random.shuffle(init_times_unix_sec)
                init_time_index = 0

            this_target_matrix = _read_targets_one_example(
                init_time_unix_sec=init_times_unix_sec[init_time_index],
                target_lead_time_hours=target_lead_time_hours,
                target_field_names=target_field_names,
                target_dir_name=target_dir_name,
                target_norm_param_table_xarray=target_norm_param_table_xarray,
                use_quantile_norm=targets_use_quantile_norm,
                subset_grid=subset_grid
            )

            if this_target_matrix is None:
                init_time_index += 1
                continue

            i = num_examples_in_memory + 0
            target_matrix[i, ...] = this_target_matrix

            if do_residual_prediction:
                this_baseline_matrix = _read_residual_baseline_one_example(
                    init_time_unix_sec=init_times_unix_sec[init_time_index],
                    nwp_model_name=resid_baseline_model_name,
                    nwp_lead_time_hours=resid_baseline_lead_time_hours,
                    nwp_directory_name=resid_baseline_model_dir_name,
                    target_field_names=target_field_names,
                    subset_grid=subset_grid,
                    predict_dewpoint_depression=predict_dewpoint_depression,
                    predict_gust_factor=predict_gust_factor
                )

                if this_baseline_matrix is None:
                    init_time_index += 1
                    continue

                predictor_matrix_resid_baseline[i, ...] = this_baseline_matrix

            (
                this_predictor_matrix_2pt5km,
                this_predictor_matrix_10km,
                this_predictor_matrix_20km,
                this_predictor_matrix_40km
            ) = _read_predictors_one_example(
                init_time_unix_sec=init_times_unix_sec[init_time_index],
                nwp_model_names=nwp_model_names,
                nwp_lead_times_hours=nwp_lead_times_hours,
                nwp_model_to_field_names=nwp_model_to_field_names,
                nwp_model_to_dir_name=nwp_model_to_dir_name,
                nwp_norm_param_table_xarray=nwp_norm_param_table_xarray,
                use_quantile_norm=nwp_use_quantile_norm,
                subset_grid=subset_grid
            )

            found_any_predictors = True

            if predictor_matrix_2pt5km is not None:
                found_any_predictors &= not numpy.all(
                    numpy.isnan(this_predictor_matrix_2pt5km)
                )
                predictor_matrix_2pt5km[i, ...] = this_predictor_matrix_2pt5km
            if predictor_matrix_10km is not None:
                found_any_predictors &= not numpy.all(
                    numpy.isnan(this_predictor_matrix_10km)
                )
                predictor_matrix_10km[i, ...] = this_predictor_matrix_10km
            if predictor_matrix_20km is not None:
                found_any_predictors &= not numpy.all(
                    numpy.isnan(this_predictor_matrix_20km)
                )
                predictor_matrix_20km[i, ...] = this_predictor_matrix_20km
            if predictor_matrix_40km is not None:
                found_any_predictors &= not numpy.all(
                    numpy.isnan(this_predictor_matrix_40km)
                )
                predictor_matrix_40km[i, ...] = this_predictor_matrix_40km

            if not found_any_predictors:
                init_time_index += 1
                continue

            num_examples_in_memory += 1
            init_time_index += 1

        error_checking.assert_is_numpy_array_without_nan(target_matrix)
        print((
            'Shape of 2.5-km target matrix and NaN fraction: '
            '{0:s}, {1:.04f}'
        ).format(
            str(target_matrix.shape),
            numpy.mean(numpy.isnan(target_matrix))
        ))

        if predictor_matrix_2pt5km is not None:
            print((
                'Shape of 2.5-km predictor matrix and NaN fraction: '
                '{0:s}, {1:.04f}'
            ).format(
                str(predictor_matrix_2pt5km.shape),
                numpy.mean(numpy.isnan(predictor_matrix_2pt5km))
            ))

            predictor_matrix_2pt5km[numpy.isnan(predictor_matrix_2pt5km)] = (
                sentinel_value
            )

        if nbm_constant_matrix is not None:
            print('Shape of NBM-constant predictor matrix: {0:s}'.format(
                str(nbm_constant_matrix.shape)
            ))

        if predictor_matrix_resid_baseline is not None:
            print((
                'Shape of residual baseline matrix and NaN fraction: '
                '{0:s}, {1:.04f}'
            ).format(
                str(predictor_matrix_resid_baseline.shape),
                numpy.mean(numpy.isnan(predictor_matrix_resid_baseline))
            ))

            print('Min values in residual baseline matrix: {0:s}'.format(
                str(numpy.nanmin(predictor_matrix_resid_baseline, axis=(0, 1, 2)))
            ))
            print('Max values in residual baseline matrix: {0:s}'.format(
                str(numpy.nanmax(predictor_matrix_resid_baseline, axis=(0, 1, 2)))
            ))

            error_checking.assert_is_numpy_array_without_nan(
                predictor_matrix_resid_baseline
            )

        if predictor_matrix_10km is not None:
            print((
                'Shape of 10-km predictor matrix and NaN fraction: '
                '{0:s}, {1:.04f}'
            ).format(
                str(predictor_matrix_10km.shape),
                numpy.mean(numpy.isnan(predictor_matrix_10km))
            ))

            predictor_matrix_10km[numpy.isnan(predictor_matrix_10km)] = (
                sentinel_value
            )

        if predictor_matrix_20km is not None:
            print((
                'Shape of 20-km predictor matrix and NaN fraction: '
                '{0:s}, {1:.04f}'
            ).format(
                str(predictor_matrix_20km.shape),
                numpy.mean(numpy.isnan(predictor_matrix_20km))
            ))

            predictor_matrix_20km[numpy.isnan(predictor_matrix_20km)] = (
                sentinel_value
            )

        if predictor_matrix_40km is not None:
            print((
                'Shape of 40-km predictor matrix and NaN fraction: '
                '{0:s}, {1:.04f}'
            ).format(
                str(predictor_matrix_40km.shape),
                numpy.mean(numpy.isnan(predictor_matrix_40km))
            ))

            predictor_matrix_40km[numpy.isnan(predictor_matrix_40km)] = (
                sentinel_value
            )

        predictor_matrices = {}
        if predictor_matrix_2pt5km is not None:
            predictor_matrices.update({
                '2pt5km_inputs': predictor_matrix_2pt5km.astype('float32')
            })
        if nbm_constant_matrix is not None:
            predictor_matrices.update({
                'const_inputs': nbm_constant_matrix.astype('float32')
            })
        if predictor_matrix_10km is not None:
            predictor_matrices.update({
                '10km_inputs': predictor_matrix_10km.astype('float32')
            })
        if predictor_matrix_20km is not None:
            predictor_matrices.update({
                '20km_inputs': predictor_matrix_20km.astype('float32')
            })
        if predictor_matrix_40km is not None:
            predictor_matrices.update({
                '40km_inputs': predictor_matrix_40km.astype('float32')
            })
        if predictor_matrix_resid_baseline is not None:
            predictor_matrices.update({
                'resid_baseline_inputs':
                    predictor_matrix_resid_baseline.astype('float32')
            })

        yield predictor_matrices, target_matrix


def train_model(
        model_object, num_epochs,
        num_training_batches_per_epoch, training_option_dict,
        num_validation_batches_per_epoch, validation_option_dict,
        loss_function_string, optimizer_function_string,
        metric_function_strings, plateau_patience_epochs,
        plateau_learning_rate_multiplier, early_stopping_patience_epochs,
        output_dir_name):
    """Trains neural net with generator.

    :param model_object: Untrained neural net (instance of
        `keras.models.Model`).
    :param num_epochs: Number of training epochs.
    :param num_training_batches_per_epoch: Number of training batches per epoch.
    :param training_option_dict: See doc for `data_generator`.  This dictionary
        will be used to generate training data.
    :param num_validation_batches_per_epoch: Number of validation batches per
        epoch.
    :param validation_option_dict: See doc for `data_generator`.  For validation
        only, the following values will replace corresponding values in
        `training_option_dict`:
    validation_option_dict["first_init_times_unix_sec"]
    validation_option_dict["last_init_times_unix_sec"]
    validation_option_dict["nwp_model_to_dir_name"]
    validation_option_dict["target_dir_name"]

    :param loss_function_string: Loss function.  This string should be formatted
        such that `eval(loss_function_string)` returns the actual loss function.
    :param optimizer_function_string: Optimizer.  This string should be
        formatted such that `eval(optimizer_function_string)` returns the actual
        optimizer.
    :param metric_function_strings: 1-D list with names of metrics.  Each string
        should be formatted such that `eval(metric_function_strings[i])` returns
        the actual metric function.
    :param plateau_patience_epochs: Training will be deemed to have reached
        "plateau" if validation loss has not decreased in the last N epochs,
        where N = plateau_patience_epochs.
    :param plateau_learning_rate_multiplier: If training reaches "plateau,"
        learning rate will be multiplied by this value in range (0, 1).
    :param early_stopping_patience_epochs: Training will be stopped early if
        validation loss has not decreased in the last N epochs, where N =
        early_stopping_patience_epochs.
    :param output_dir_name: Path to output directory (model and training history
        will be saved here).
    """

    file_system_utils.mkdir_recursive_if_necessary(
        directory_name=output_dir_name
    )

    backup_dir_name = '{0:s}/backup_and_restore'.format(output_dir_name)
    file_system_utils.mkdir_recursive_if_necessary(
        directory_name=backup_dir_name
    )

    error_checking.assert_is_integer(num_epochs)
    error_checking.assert_is_geq(num_epochs, 2)
    error_checking.assert_is_integer(num_training_batches_per_epoch)
    error_checking.assert_is_geq(num_training_batches_per_epoch, 2)
    error_checking.assert_is_integer(num_validation_batches_per_epoch)
    error_checking.assert_is_geq(num_validation_batches_per_epoch, 2)
    error_checking.assert_is_integer(plateau_patience_epochs)
    error_checking.assert_is_geq(plateau_patience_epochs, 2)
    error_checking.assert_is_greater(plateau_learning_rate_multiplier, 0.)
    error_checking.assert_is_less_than(plateau_learning_rate_multiplier, 1.)
    error_checking.assert_is_integer(early_stopping_patience_epochs)
    error_checking.assert_is_geq(early_stopping_patience_epochs, 5)

    validation_keys_to_keep = [
        FIRST_INIT_TIMES_KEY, LAST_INIT_TIMES_KEY,
        NWP_MODEL_TO_DIR_KEY, TARGET_DIR_KEY
    ]
    for this_key in list(training_option_dict.keys()):
        if this_key in validation_keys_to_keep:
            continue

        validation_option_dict[this_key] = training_option_dict[this_key]

    training_option_dict = _check_generator_args(training_option_dict)
    validation_option_dict = _check_generator_args(validation_option_dict)

    model_file_name = '{0:s}/model.keras'.format(output_dir_name)

    history_object = keras.callbacks.CSVLogger(
        filename='{0:s}/history.csv'.format(output_dir_name),
        separator=',', append=False
    )
    checkpoint_object = keras.callbacks.ModelCheckpoint(
        filepath=model_file_name, monitor='val_loss', verbose=1,
        save_best_only=True, save_weights_only=False, mode='min',
        save_freq='epoch'
    )
    early_stopping_object = keras.callbacks.EarlyStopping(
        monitor='val_loss', min_delta=0.,
        patience=early_stopping_patience_epochs, verbose=1, mode='min'
    )
    plateau_object = keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=plateau_learning_rate_multiplier,
        patience=plateau_patience_epochs, verbose=1, mode='min',
        min_delta=0., cooldown=0
    )
    backup_object = keras.callbacks.BackupAndRestore(
        backup_dir_name, save_freq='epoch', delete_checkpoint=True
    )

    list_of_callback_objects = [
        history_object, checkpoint_object,
        early_stopping_object, plateau_object,
        backup_object
    ]

    training_generator = data_generator(training_option_dict)
    validation_generator = data_generator(validation_option_dict)

    metafile_name = find_metafile(
        model_file_name=model_file_name, raise_error_if_missing=False
    )
    print('Writing metadata to: "{0:s}"...'.format(metafile_name))

    write_metafile(
        pickle_file_name=metafile_name,
        num_epochs=num_epochs,
        num_training_batches_per_epoch=num_training_batches_per_epoch,
        training_option_dict=training_option_dict,
        num_validation_batches_per_epoch=num_validation_batches_per_epoch,
        validation_option_dict=validation_option_dict,
        loss_function_string=loss_function_string,
        optimizer_function_string=optimizer_function_string,
        metric_function_strings=metric_function_strings,
        plateau_patience_epochs=plateau_patience_epochs,
        plateau_learning_rate_multiplier=plateau_learning_rate_multiplier,
        early_stopping_patience_epochs=early_stopping_patience_epochs
    )

    model_object.fit(
        x=training_generator,
        steps_per_epoch=num_training_batches_per_epoch,
        epochs=num_epochs, verbose=1, callbacks=list_of_callback_objects,
        validation_data=validation_generator,
        validation_steps=num_validation_batches_per_epoch
    )


def apply_model(
        model_object, predictor_matrices, num_examples_per_batch,
        predict_dewpoint_depression, predict_gust_factor,
        verbose=True, target_field_names=None):
    """Applies trained neural net -- inference time!

    E = number of examples
    M = number of rows in grid
    N = number of columns in grid
    F = number of target fields

    :param model_object: Trained neural net (instance of `keras.models.Model`).
    :param predictor_matrices: See output doc for `data_generator`.
    :param num_examples_per_batch: Batch size.
    :param predict_dewpoint_depression: Boolean flag.  If True, the NN predicts
        dewpoint depression, which will be converted to dewpoint temperature.
    :param predict_gust_factor: Boolean flag.  If True, the NN predicts gust
        factor, which will be converted to gust speed.
    :param verbose: Boolean flag.  If True, will print progress messages.
    :param target_field_names:
        [used only if predict_dewpoint_depression or predict_gust_factor]
        length-F list of target fields (each must be accepted by
        `urma_utils.check_field_name`).
    :return: prediction_matrix: E-by-M-by-N-by-F numpy array of predicted
        values.
    """

    # Check input args.
    for this_matrix in predictor_matrices:
        error_checking.assert_is_numpy_array_without_nan(this_matrix)

    error_checking.assert_is_integer(num_examples_per_batch)
    error_checking.assert_is_geq(num_examples_per_batch, 1)
    num_examples = predictor_matrices[0].shape[0]
    num_examples_per_batch = min([num_examples_per_batch, num_examples])

    error_checking.assert_is_boolean(predict_dewpoint_depression)
    error_checking.assert_is_boolean(predict_gust_factor)
    error_checking.assert_is_boolean(verbose)

    if predict_dewpoint_depression or predict_gust_factor:
        error_checking.assert_is_string_list(target_field_names)
        for this_name in target_field_names:
            urma_utils.check_field_name(this_name)

    # Do actual stuff.
    prediction_matrix = None

    for i in range(0, num_examples, num_examples_per_batch):
        first_index = i
        last_index = min([i + num_examples_per_batch, num_examples])

        if verbose:
            print('Applying model to examples {0:d}-{1:d} of {2:d}...'.format(
                first_index + 1, last_index, num_examples
            ))

        this_prediction_matrix = model_object.predict_on_batch(
            [a[first_index:last_index, ...] for a in predictor_matrices]
        )

        if prediction_matrix is None:
            dimensions = (num_examples,) + this_prediction_matrix.shape[1:]
            prediction_matrix = numpy.full(dimensions, numpy.nan)

        prediction_matrix[first_index:last_index, ...] = this_prediction_matrix

    if verbose:
        print('Have applied model to all {0:d} examples!'.format(num_examples))

    while len(prediction_matrix.shape) < 4:
        prediction_matrix = numpy.expand_dims(prediction_matrix, axis=-1)

    if predict_dewpoint_depression:
        prediction_matrix = __predicted_2m_depression_to_dewpoint(
            prediction_matrix=prediction_matrix,
            target_field_names=target_field_names
        )

    if predict_gust_factor:
        prediction_matrix = __predicted_10m_gust_factor_to_speed(
            prediction_matrix=prediction_matrix,
            target_field_names=target_field_names
        )

    return prediction_matrix


def find_metafile(model_file_name, raise_error_if_missing=True):
    """Finds metafile for neural net.

    :param model_file_name: Path to model file.
    :param raise_error_if_missing: Boolean flag.  If file is missing and
        `raise_error_if_missing == True`, will throw error.  If file is missing
        and `raise_error_if_missing == False`, will return *expected* file path.
    :return: metafile_name: Path to metafile.
    """

    error_checking.assert_is_string(model_file_name)
    metafile_name = '{0:s}/model_metadata.p'.format(
        os.path.split(model_file_name)[0]
    )

    if raise_error_if_missing and not os.path.isfile(metafile_name):
        error_string = 'Cannot find file.  Expected at: "{0:s}"'.format(
            metafile_name
        )
        raise ValueError(error_string)

    return metafile_name


def write_metafile(
        pickle_file_name, num_epochs, num_training_batches_per_epoch,
        training_option_dict, num_validation_batches_per_epoch,
        validation_option_dict, loss_function_string, optimizer_function_string,
        metric_function_strings, plateau_patience_epochs,
        plateau_learning_rate_multiplier, early_stopping_patience_epochs):
    """Writes metadata to Pickle file.

    :param pickle_file_name: Path to output file.
    :param num_epochs: See doc for `train_model`.
    :param num_training_batches_per_epoch: Same.
    :param training_option_dict: Same.
    :param num_validation_batches_per_epoch: Same.
    :param validation_option_dict: Same.
    :param loss_function_string: Same.
    :param optimizer_function_string: Same.
    :param metric_function_strings: Same.
    :param plateau_patience_epochs: Same.
    :param plateau_learning_rate_multiplier: Same.
    :param early_stopping_patience_epochs: Same.
    """

    metadata_dict = {
        NUM_EPOCHS_KEY: num_epochs,
        NUM_TRAINING_BATCHES_KEY: num_training_batches_per_epoch,
        TRAINING_OPTIONS_KEY: training_option_dict,
        NUM_VALIDATION_BATCHES_KEY: num_validation_batches_per_epoch,
        VALIDATION_OPTIONS_KEY: validation_option_dict,
        LOSS_FUNCTION_KEY: loss_function_string,
        OPTIMIZER_FUNCTION_KEY: optimizer_function_string,
        METRIC_FUNCTIONS_KEY: metric_function_strings,
        PLATEAU_PATIENCE_KEY: plateau_patience_epochs,
        PLATEAU_LR_MUTIPLIER_KEY: plateau_learning_rate_multiplier,
        EARLY_STOPPING_PATIENCE_KEY: early_stopping_patience_epochs
    }

    file_system_utils.mkdir_recursive_if_necessary(file_name=pickle_file_name)

    pickle_file_handle = open(pickle_file_name, 'wb')
    pickle.dump(metadata_dict, pickle_file_handle)
    pickle_file_handle.close()


def read_metafile(pickle_file_name):
    """Reads metadata from Pickle file.

    :param pickle_file_name: Path to input file.
    :return: metadata_dict: Dictionary with the following keys.
    metadata_dict["num_epochs"]: See doc for `train_model`.
    metadata_dict["num_training_batches_per_epoch"]: Same.
    metadata_dict["training_option_dict"]: Same.
    metadata_dict["num_validation_batches_per_epoch"]: Same.
    metadata_dict["validation_option_dict"]: Same.
    metadata_dict["loss_function_string"]: Same.
    metadata_dict["optimizer_function_string"]: Same.
    metadata_dict["metric_function_strings"]: Same.
    metadata_dict["plateau_patience_epochs"]: Same.
    metadata_dict["plateau_learning_rate_multiplier"]: Same.
    metadata_dict["early_stopping_patience_epochs"]: Same.

    :raises: ValueError: if any expected key is not found in dictionary.
    """

    error_checking.assert_file_exists(pickle_file_name)

    pickle_file_handle = open(pickle_file_name, 'rb')
    metadata_dict = pickle.load(pickle_file_handle)
    pickle_file_handle.close()

    missing_keys = list(set(METADATA_KEYS) - set(metadata_dict.keys()))
    if len(missing_keys) == 0:
        return metadata_dict

    error_string = (
        '\n{0:s}\nKeys listed above were expected, but not found, in file '
        '"{1:s}".'
    ).format(str(missing_keys), pickle_file_name)

    raise ValueError(error_string)


def read_model(hdf5_file_name):
    """Reads model from HDF5 file.

    :param hdf5_file_name: Path to input file.
    :return: model_object: Instance of `keras.models.Model`.
    """

    error_checking.assert_file_exists(hdf5_file_name)

    metafile_name = find_metafile(
        model_file_name=hdf5_file_name, raise_error_if_missing=True
    )
    metadata_dict = read_metafile(metafile_name)

    print(metadata_dict[LOSS_FUNCTION_KEY])
    custom_object_dict = {
        'loss': eval(metadata_dict[LOSS_FUNCTION_KEY])
    }
    model_object = load_model(
        hdf5_file_name, custom_objects=custom_object_dict, compile=False
    )

    metric_function_list = [
        eval(m) for m in metadata_dict[METRIC_FUNCTIONS_KEY]
    ]
    model_object.compile(
        loss=custom_object_dict['loss'],
        optimizer=eval(metadata_dict[OPTIMIZER_FUNCTION_KEY]),
        metrics=metric_function_list
    )

    return model_object

"""Methods for patchwise NN-training."""

import os
import sys
import pickle
import warnings
import numpy
import keras
from scipy.interpolate import interp1d
from tensorflow.keras.saving import load_model

THIS_DIRECTORY_NAME = os.path.dirname(os.path.realpath(
    os.path.join(os.getcwd(), os.path.expanduser(__file__))
))
sys.path.append(os.path.normpath(os.path.join(THIS_DIRECTORY_NAME, '..')))

import time_conversion
import time_periods
import number_rounding
import temperature_conversions as temperature_conv
import file_system_utils
import error_checking
import nwp_model_io
import interp_nwp_model_io
import nbm_constant_io
import urma_io
import nbm_utils
import misc_utils
import nwp_model_utils
import urma_utils
import nbm_constant_utils
import normalization
import custom_losses
import custom_metrics

TIME_FORMAT = '%Y-%m-%d-%H'
HOURS_TO_SECONDS = 3600
DEFAULT_GUST_FACTOR = 1.5

POSSIBLE_GRID_SPACINGS_KM = numpy.array([2.5, 10, 20, 40])
POSSIBLE_DOWNSAMPLING_FACTORS = numpy.array([1, 4, 8, 16], dtype=int)

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
PATCH_SIZE_KEY = 'patch_size_2pt5km_pixels'

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
LATITUDE_MATRIX_KEY = 'latitude_matrix_deg_n'
LONGITUDE_MATRIX_KEY = 'longitude_matrix_deg_e'

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
PATCH_OVERLAP_FOR_FAST_GEN_KEY = 'patch_overlap_fast_gen_2pt5km_pixels'

METADATA_KEYS = [
    NUM_EPOCHS_KEY, NUM_TRAINING_BATCHES_KEY, TRAINING_OPTIONS_KEY,
    NUM_VALIDATION_BATCHES_KEY, VALIDATION_OPTIONS_KEY, LOSS_FUNCTION_KEY,
    OPTIMIZER_FUNCTION_KEY, METRIC_FUNCTIONS_KEY, PLATEAU_PATIENCE_KEY,
    PLATEAU_LR_MUTIPLIER_KEY, EARLY_STOPPING_PATIENCE_KEY,
    PATCH_OVERLAP_FOR_FAST_GEN_KEY
]

NUM_FULL_ROWS_KEY = 'num_rows_in_full_grid'
NUM_FULL_COLUMNS_KEY = 'num_columns_in_full_grid'
NUM_PATCH_ROWS_KEY = 'num_rows_in_patch'
NUM_PATCH_COLUMNS_KEY = 'num_columns_in_patch'
PATCH_OVERLAP_SIZE_KEY = 'patch_overlap_size_2pt5km_pixels'
PATCH_START_ROW_KEY = 'patch_start_row_2pt5km'
PATCH_START_COLUMN_KEY = 'patch_start_column_2pt5km'


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


def __interp_predictors_by_lead_time(predictor_matrix, lead_times_hours):
    """Interpolates predictors to fill missing lead times.

    M = number of rows in grid
    N = number of columns in grid
    L = number of lead times
    P = number of predictor variables

    :param predictor_matrix: M-by-N-by-L-by-P numpy array of predictor values.
    :param lead_times_hours: length-L numpy array of lead times.
    :return: predictor_matrix: Same as input, though maybe with fewer NaN's.
    """

    num_predictors = predictor_matrix.shape[-1]

    for p in range(num_predictors):
        missing_lead_time_flags = numpy.all(
            numpy.isnan(predictor_matrix[..., p]),
            axis=(0, 1)
        )
        missing_lead_time_indices = numpy.where(missing_lead_time_flags)[0]
        filled_lead_time_indices = numpy.where(
            numpy.invert(missing_lead_time_flags)
        )[0]

        if len(filled_lead_time_indices) == 0:
            continue

        missing_lead_time_indices = [
            k for k in missing_lead_time_indices if
            k >= numpy.min(filled_lead_time_indices)
        ]
        missing_lead_time_indices = [
            k for k in missing_lead_time_indices if
            k < numpy.max(filled_lead_time_indices)
        ]

        if len(missing_lead_time_indices) == 0:
            continue

        interp_object = interp1d(
            x=lead_times_hours[filled_lead_time_indices],
            y=predictor_matrix[..., filled_lead_time_indices, p],
            axis=2,
            kind='linear',
            bounds_error=True,
            assume_sorted=True
        )

        predictor_matrix[..., missing_lead_time_indices, p] = interp_object(
            lead_times_hours[missing_lead_time_indices]
        )

    return predictor_matrix


def __get_grid_dimensions(grid_spacing_km, patch_location_dict):
    """Determines grid dimensions for neural network.

    :param grid_spacing_km: Grid spacing (must be 2.5, 10, 20, or 40 km).
    :param patch_location_dict: Dictionary produced by
        `misc_utils.determine_patch_locations`.  If you are training with the
        full grid (not the patchwise approach), make this None.
    :return: num_rows: Number of rows in grid.
    :return: num_columns: Number of columns in grid.
    """

    k = numpy.argmin(
        numpy.absolute(grid_spacing_km - POSSIBLE_GRID_SPACINGS_KM)
    )

    if k == 0:
        if patch_location_dict is None:
            num_rows, num_columns = nwp_model_utils.model_to_nbm_grid_size(
                nwp_model_utils.HRRR_MODEL_NAME
            )
        else:
            num_rows = numpy.diff(
                patch_location_dict[misc_utils.ROW_LIMITS_2PT5KM_KEY]
            )[0] + 1

            num_columns = numpy.diff(
                patch_location_dict[misc_utils.COLUMN_LIMITS_2PT5KM_KEY]
            )[0] + 1

        return num_rows, num_columns

    if k == 1:
        if patch_location_dict is None:
            num_rows, num_columns = nwp_model_utils.model_to_nbm_grid_size(
                nwp_model_utils.RAP_MODEL_NAME
            )
        else:
            num_rows = numpy.diff(
                patch_location_dict[misc_utils.ROW_LIMITS_10KM_KEY]
            )[0] + 1

            num_columns = numpy.diff(
                patch_location_dict[misc_utils.COLUMN_LIMITS_10KM_KEY]
            )[0] + 1

        return num_rows, num_columns

    if k == 2:
        if patch_location_dict is None:
            num_rows, num_columns = nwp_model_utils.model_to_nbm_grid_size(
                nwp_model_utils.GFS_MODEL_NAME
            )
        else:
            num_rows = numpy.diff(
                patch_location_dict[misc_utils.ROW_LIMITS_20KM_KEY]
            )[0] + 1

            num_columns = numpy.diff(
                patch_location_dict[misc_utils.COLUMN_LIMITS_20KM_KEY]
            )[0] + 1

        return num_rows, num_columns

    if patch_location_dict is None:
        num_rows, num_columns = nwp_model_utils.model_to_nbm_grid_size(
            nwp_model_utils.GEFS_MODEL_NAME
        )
    else:
        num_rows = numpy.diff(
            patch_location_dict[misc_utils.ROW_LIMITS_40KM_KEY]
        )[0] + 1

        num_columns = numpy.diff(
            patch_location_dict[misc_utils.COLUMN_LIMITS_40KM_KEY]
        )[0] + 1

    return num_rows, num_columns


def __update_patch_metalocation_dict(patch_metalocation_dict):
    """Updates patch-metalocation dictionary.

    This is fancy talk for "determines where the next patch will be, when
    applying a patchwise-trained neural net over the full grid".

    :param patch_metalocation_dict: Dictionary with the following keys.
    patch_metalocation_dict["num_rows_in_full_grid"]: Number of rows in full
        grid.
    patch_metalocation_dict["num_columns_in_full_grid"]: Number of columns in
        full grid.
    patch_metalocation_dict["num_rows_in_patch"]: Number of rows in each patch.
    patch_metalocation_dict["num_columns_in_patch"]: Number of columns in each
        patch.
    patch_metalocation_dict["patch_overlap_size_2pt5km_pixels"]: Overlap between
        adjacent patches, in terms of pixels at the finest resolution (2.5 km).
        All other values in this dictionary are in terms of the finest
        resolution as well, so in hindsight I guess it's weird that this key
        specifies "2pt5km" in the variable name while the others don't.  Meh.
    patch_metalocation_dict["patch_start_row_2pt5km"]: First row covered by the
        current patch location, in the finest-resolution (2.5-km) grid.
    patch_metalocation_dict["patch_start_column_2pt5km"]: First column covered
        by the current patch location, in the finest-resolution (2.5-km) grid.

    :return: patch_metalocation_dict: Same as input, except that keys
        "patch_start_row_2pt5km" and "patch_start_column_2pt5km" have been
        updated.
    """

    pmld = patch_metalocation_dict
    num_rows_in_full_grid = pmld[NUM_FULL_ROWS_KEY]
    num_columns_in_full_grid = pmld[NUM_FULL_COLUMNS_KEY]
    num_rows_in_patch = pmld[NUM_PATCH_ROWS_KEY]
    num_columns_in_patch = pmld[NUM_PATCH_COLUMNS_KEY]
    patch_overlap_size_2pt5km_pixels = pmld[PATCH_OVERLAP_SIZE_KEY]
    patch_end_row_2pt5km = pmld[PATCH_START_ROW_KEY] + num_rows_in_patch - 1
    patch_end_column_2pt5km = (
        pmld[PATCH_START_COLUMN_KEY] + num_columns_in_patch - 1
    )

    # if pmld[PATCH_START_ROW_KEY] < 0:
    #     patch_end_row_2pt5km = num_rows_in_patch - 1
    #     patch_end_column_2pt5km = num_columns_in_patch - 1
    # elif patch_end_column_2pt5km == num_columns_in_full_grid - 1:
    #     if patch_end_row_2pt5km == num_rows_in_full_grid - 1:
    #         patch_end_row_2pt5km = -1
    #         patch_end_column_2pt5km = -1
    #     else:
    #         patch_end_row_2pt5km += (
    #             num_rows_in_patch - 2 * patch_overlap_size_2pt5km_pixels
    #         )
    #         patch_end_column_2pt5km = num_columns_in_patch - 1
    # else:
    #     patch_end_column_2pt5km += (
    #         num_columns_in_patch - 2 * patch_overlap_size_2pt5km_pixels
    #     )

    if pmld[PATCH_START_ROW_KEY] < 0:
        patch_end_row_2pt5km = num_rows_in_patch - 1
        patch_end_column_2pt5km = num_columns_in_patch - 1
    elif patch_end_column_2pt5km >= num_columns_in_full_grid - 16:
        if patch_end_row_2pt5km >= num_rows_in_full_grid - 16:
            patch_end_row_2pt5km = -1
            patch_end_column_2pt5km = -1
        else:
            patch_end_row_2pt5km += (
                num_rows_in_patch - 2 * patch_overlap_size_2pt5km_pixels
            )
            patch_end_column_2pt5km = num_columns_in_patch - 1
    else:
        patch_end_column_2pt5km += (
            num_columns_in_patch - 2 * patch_overlap_size_2pt5km_pixels
        )

    patch_end_row_2pt5km = min([
        patch_end_row_2pt5km, num_rows_in_full_grid - 1
    ])
    patch_end_column_2pt5km = min([
        patch_end_column_2pt5km, num_columns_in_full_grid - 1
    ])

    # TODO(thunderhoser): I might want to be more flexible about this whole
    # divisible-by-16 thing.  It assumes that I have inputs at both 2.5-km
    # resolution (which I always do) and 40-km resolution (which I don't
    # always).
    patch_start_row_2pt5km = patch_end_row_2pt5km - num_rows_in_patch + 1
    patch_start_row_2pt5km = number_rounding.floor_to_nearest(
        patch_start_row_2pt5km, 16
    )
    patch_start_row_2pt5km = numpy.round(patch_start_row_2pt5km).astype(int)

    patch_start_column_2pt5km = (
        patch_end_column_2pt5km - num_columns_in_patch + 1
    )
    patch_start_column_2pt5km = number_rounding.floor_to_nearest(
        patch_start_column_2pt5km, 16
    )
    patch_start_column_2pt5km = (
        numpy.round(patch_start_column_2pt5km).astype(int)
    )

    pmld[PATCH_START_ROW_KEY] = patch_start_row_2pt5km
    pmld[PATCH_START_COLUMN_KEY] = patch_start_column_2pt5km
    patch_metalocation_dict = pmld

    return patch_metalocation_dict


def __increment_init_time(current_index, init_times_unix_sec):
    """Increments initialization time for generator.

    This allows the generator to read the next init time.

    :param current_index: Current index.  If current_index == k, this means the
        last init time read is init_times_unix_sec[k].
    :param init_times_unix_sec: 1-D numpy array of init times.
    :return: current_index: Updated version of input.
    :return: init_times_unix_sec: Possibly shuffled version of input.
    """

    if current_index == len(init_times_unix_sec) - 1:
        numpy.random.shuffle(init_times_unix_sec)
        current_index = 0
    else:
        current_index += 1

    return current_index, init_times_unix_sec


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

    nwp_model_names = second_nwp_model_names
    nwp_model_names.sort()

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

    if option_dict[PATCH_SIZE_KEY] is not None:
        error_checking.assert_is_integer(option_dict[PATCH_SIZE_KEY])

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
        patch_location_dict):
    """Initializes predictor matrices for one example.

    :param nwp_model_names: 1-D list with names of NWP models.
    :param nwp_model_to_field_names: Dictionary.  For details, see documentation
        for `data_generator`.
    :param num_nwp_lead_times: Number of lead times.
    :param patch_location_dict: Dictionary produced by
        `misc_utils.determine_patch_locations`.  If you are training with the
        full grid (not the patchwise approach), make this None.
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
    num_rows, num_columns = __get_grid_dimensions(
        grid_spacing_km=2.5,
        patch_location_dict=patch_location_dict
    )

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
    num_rows, num_columns = __get_grid_dimensions(
        grid_spacing_km=10.,
        patch_location_dict=patch_location_dict
    )

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
    num_rows, num_columns = __get_grid_dimensions(
        grid_spacing_km=20.,
        patch_location_dict=patch_location_dict
    )

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
    num_rows, num_columns = __get_grid_dimensions(
        grid_spacing_km=40.,
        patch_location_dict=patch_location_dict
    )

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
        num_target_fields, num_examples_per_batch, do_residual_prediction,
        patch_location_dict):
    """Initializes predictor and target matrices for one batch.

    :param nwp_model_names: 1-D list with names of NWP models.
    :param nwp_model_to_field_names: Dictionary.  For details, see documentation
        for `data_generator`.
    :param num_nwp_lead_times: Number of lead times.
    :param num_target_fields: Number of target fields.
    :param num_examples_per_batch: Batch size.
    :param do_residual_prediction: Boolean flag.  If True, the NN is predicting
        difference between a given NWP forecast and the URMA truth.  If True,
        the NN is predicting the URMA truth directly.
    :param patch_location_dict: Dictionary produced by
        `misc_utils.determine_patch_locations`.  If you are training with the
        full grid (not the patchwise approach), make this None.
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
    num_rows, num_columns = __get_grid_dimensions(
        grid_spacing_km=2.5,
        patch_location_dict=patch_location_dict
    )

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
    num_rows, num_columns = __get_grid_dimensions(
        grid_spacing_km=10.,
        patch_location_dict=patch_location_dict
    )

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
    num_rows, num_columns = __get_grid_dimensions(
        grid_spacing_km=20.,
        patch_location_dict=patch_location_dict
    )

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
    num_rows, num_columns = __get_grid_dimensions(
        grid_spacing_km=40.,
        patch_location_dict=patch_location_dict
    )

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
        target_norm_param_table_xarray, use_quantile_norm, patch_location_dict):
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
    :param patch_location_dict: Dictionary produced by
        `misc_utils.determine_patch_locations`.  If you are training with the
        full grid (not the patchwise approach), make this None.
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
        raise_error_if_missing=False
    )

    if not os.path.isfile(urma_file_name):
        warning_string = (
            'POTENTIAL ERROR: Could not find file expected at: "{0:s}"'
        ).format(urma_file_name)

        warnings.warn(warning_string)
        return None

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

    if patch_location_dict is not None:
        i_start = patch_location_dict[misc_utils.ROW_LIMITS_2PT5KM_KEY][0]
        i_end = patch_location_dict[misc_utils.ROW_LIMITS_2PT5KM_KEY][1] + 1
        j_start = patch_location_dict[misc_utils.COLUMN_LIMITS_2PT5KM_KEY][0]
        j_end = patch_location_dict[misc_utils.COLUMN_LIMITS_2PT5KM_KEY][1] + 1

        target_matrix = target_matrix[i_start:i_end, j_start:j_end]

    return target_matrix


def _read_residual_baseline_one_example(
        init_time_unix_sec, nwp_model_name, nwp_lead_time_hours,
        nwp_directory_name, target_field_names, patch_location_dict,
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
    :param patch_location_dict: Dictionary produced by
        `misc_utils.determine_patch_locations`.  If you are training with the
        full grid (not the patchwise approach), make this None.
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

    if patch_location_dict is not None:
        i_start = patch_location_dict[misc_utils.ROW_LIMITS_2PT5KM_KEY][0]
        i_end = patch_location_dict[misc_utils.ROW_LIMITS_2PT5KM_KEY][1]
        j_start = patch_location_dict[misc_utils.COLUMN_LIMITS_2PT5KM_KEY][0]
        j_end = patch_location_dict[misc_utils.COLUMN_LIMITS_2PT5KM_KEY][1]

        nwp_forecast_table_xarray = nwp_model_utils.subset_by_row(
            nwp_forecast_table_xarray=nwp_forecast_table_xarray,
            desired_row_indices=numpy.linspace(
                i_start, i_end, num=i_end - i_start + 1, dtype=int
            )
        )

        nwp_forecast_table_xarray = nwp_model_utils.subset_by_column(
            nwp_forecast_table_xarray=nwp_forecast_table_xarray,
            desired_column_indices=numpy.linspace(
                j_start, j_end, num=j_end - j_start + 1, dtype=int
            )
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
        nwp_norm_param_table_xarray, use_quantile_norm, patch_location_dict):
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
    :param patch_location_dict: Dictionary produced by
        `misc_utils.determine_patch_locations`.  If you are training with the
        full grid (not the patchwise approach), make this None.
    :return: predictor_matrix_2pt5km: Same as output from `data_generator` but
        without first axis.
    :return: predictor_matrix_10km: Same as output from `data_generator` but
        without first axis.
    :return: predictor_matrix_20km: Same as output from `data_generator` but
        without first axis.
    :return: predictor_matrix_40km: Same as output from `data_generator` but
        without first axis.
    :return: found_any_predictors: Boolean flag.  If True, at least one output
        matrix contains a real value.  If False, the output matrices are all NaN
        or None.
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
        patch_location_dict=patch_location_dict
    )

    for i in range(num_nwp_models):
        for j in range(num_nwp_lead_times):
            if nwp_model_names[i] == nwp_model_utils.RAP_MODEL_NAME:
                adjusted_init_time_unix_sec = (
                    init_time_unix_sec + 3 * HOURS_TO_SECONDS
                )
                adjusted_lead_time_hours = nwp_lead_times_hours[j] - 3
                if adjusted_lead_time_hours < 1:
                    continue

                this_file_name = interp_nwp_model_io.find_file(
                    directory_name=nwp_model_to_dir_name[nwp_model_names[i]],
                    init_time_unix_sec=adjusted_init_time_unix_sec,
                    forecast_hour=adjusted_lead_time_hours,
                    model_name=nwp_model_names[i],
                    raise_error_if_missing=False
                )
            else:
                # TODO(thunderhoser): Need an input arg that dictates whether to
                # use init times every 6 hours or 12 hours.
                try:
                    nwp_model_utils.check_init_time(
                        init_time_unix_sec=init_time_unix_sec,
                        model_name=nwp_model_names[i]
                    )
                except:
                    continue

                # TODO(thunderhoser): Also need a more flexible residual
                # baseline.  If the desired model is missing, there should be a
                # priority list.
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

            if patch_location_dict is not None:
                pld = patch_location_dict

                if nwp_downsampling_factors[i] == 1:
                    i_start = pld[misc_utils.ROW_LIMITS_2PT5KM_KEY][0]
                    i_end = pld[misc_utils.ROW_LIMITS_2PT5KM_KEY][1]
                    j_start = pld[misc_utils.COLUMN_LIMITS_2PT5KM_KEY][0]
                    j_end = pld[misc_utils.COLUMN_LIMITS_2PT5KM_KEY][1]
                elif nwp_downsampling_factors[i] == 4:
                    i_start = pld[misc_utils.ROW_LIMITS_10KM_KEY][0]
                    i_end = pld[misc_utils.ROW_LIMITS_10KM_KEY][1]
                    j_start = pld[misc_utils.COLUMN_LIMITS_10KM_KEY][0]
                    j_end = pld[misc_utils.COLUMN_LIMITS_10KM_KEY][1]
                elif nwp_downsampling_factors[i] == 8:
                    i_start = pld[misc_utils.ROW_LIMITS_20KM_KEY][0]
                    i_end = pld[misc_utils.ROW_LIMITS_20KM_KEY][1]
                    j_start = pld[misc_utils.COLUMN_LIMITS_20KM_KEY][0]
                    j_end = pld[misc_utils.COLUMN_LIMITS_20KM_KEY][1]
                else:
                    i_start = pld[misc_utils.ROW_LIMITS_40KM_KEY][0]
                    i_end = pld[misc_utils.ROW_LIMITS_40KM_KEY][1]
                    j_start = pld[misc_utils.COLUMN_LIMITS_40KM_KEY][0]
                    j_end = pld[misc_utils.COLUMN_LIMITS_40KM_KEY][1]

                nwp_forecast_table_xarray = nwp_model_utils.subset_by_row(
                    nwp_forecast_table_xarray=nwp_forecast_table_xarray,
                    desired_row_indices=numpy.linspace(
                        i_start, i_end, num=i_end - i_start + 1, dtype=int
                    )
                )

                nwp_forecast_table_xarray = nwp_model_utils.subset_by_column(
                    nwp_forecast_table_xarray=nwp_forecast_table_xarray,
                    desired_column_indices=numpy.linspace(
                        j_start, j_end, num=j_end - j_start + 1, dtype=int
                    )
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

    found_any_predictors = False

    if predictor_matrices_2pt5km is None:
        predictor_matrix_2pt5km = None
    else:
        predictor_matrix_2pt5km = numpy.concatenate(
            predictor_matrices_2pt5km, axis=-1
        )
        predictor_matrix_2pt5km = __interp_predictors_by_lead_time(
            predictor_matrix=predictor_matrix_2pt5km,
            lead_times_hours=nwp_lead_times_hours
        )
        found_any_predictors |= not numpy.all(
            numpy.isnan(predictor_matrix_2pt5km)
        )

    if predictor_matrices_10km is None:
        predictor_matrix_10km = None
    else:
        predictor_matrix_10km = numpy.concatenate(
            predictor_matrices_10km, axis=-1
        )
        predictor_matrix_10km = __interp_predictors_by_lead_time(
            predictor_matrix=predictor_matrix_10km,
            lead_times_hours=nwp_lead_times_hours
        )
        found_any_predictors |= not numpy.all(
            numpy.isnan(predictor_matrix_10km)
        )

    if predictor_matrices_20km is None:
        predictor_matrix_20km = None
    else:
        predictor_matrix_20km = numpy.concatenate(
            predictor_matrices_20km, axis=-1
        )
        predictor_matrix_20km = __interp_predictors_by_lead_time(
            predictor_matrix=predictor_matrix_20km,
            lead_times_hours=nwp_lead_times_hours
        )
        found_any_predictors |= not numpy.all(
            numpy.isnan(predictor_matrix_20km)
        )

    if predictor_matrices_40km is None:
        predictor_matrix_40km = None
    else:
        predictor_matrix_40km = numpy.concatenate(
            predictor_matrices_40km, axis=-1
        )
        predictor_matrix_40km = __interp_predictors_by_lead_time(
            predictor_matrix=predictor_matrix_40km,
            lead_times_hours=nwp_lead_times_hours
        )
        found_any_predictors |= not numpy.all(
            numpy.isnan(predictor_matrix_40km)
        )

    return (
        predictor_matrix_2pt5km, predictor_matrix_10km,
        predictor_matrix_20km, predictor_matrix_40km,
        found_any_predictors
    )


def _find_relevant_init_times(first_time_by_period_unix_sec,
                              last_time_by_period_unix_sec, nwp_model_names):
    """Finds relevant model-initialization times.

    P = number of continuous periods

    :param first_time_by_period_unix_sec: length-P numpy array with start time
        of each period.
    :param last_time_by_period_unix_sec: length-P numpy array with end time
        of each period.
    :param nwp_model_names: 1-D list with names of NWP models used for predictors
        (each list item must be accepted by `nwp_model_utils.check_model_name`).
    :return: relevant_init_times_unix_sec: 1-D numpy array of relevant init
        times.
    """

    nwp_init_time_intervals_sec = numpy.array([
        nwp_model_utils.model_to_init_time_interval(m) for m in nwp_model_names
    ], dtype=int)

    slow_refresh_flags = nwp_init_time_intervals_sec > 6 * HOURS_TO_SECONDS
    if numpy.mean(slow_refresh_flags) > 0.5:
        nn_init_time_interval_sec = 12 * HOURS_TO_SECONDS
    else:
        nn_init_time_interval_sec = 6 * HOURS_TO_SECONDS

    relevant_init_times_unix_sec = numpy.concatenate([
        time_periods.range_and_interval_to_list(
            start_time_unix_sec=f,
            end_time_unix_sec=l,
            time_interval_sec=nn_init_time_interval_sec,
            include_endpoint=True
        )
        for f, l in
        zip(first_time_by_period_unix_sec, last_time_by_period_unix_sec)
    ])

    return misc_utils.remove_unused_days(relevant_init_times_unix_sec)


def create_data(option_dict, patch_start_row_2pt5km=None,
                patch_start_column_2pt5km=None):
    """Creates, instead of generates, neural-network inputs.

    E = number of examples returned
    M = number of rows in grid
    N = number of columns in grid

    :param option_dict: See documentation for `data_generator`.
    :param patch_start_row_2pt5km: [used only with patchwise approach]
        For every data sample -- i.e., every init time -- will take the patch
        from the same location, starting at this row in the 2.5-km grid.  If
        you want the patch location to move around, leave this argument alone.
    :param patch_start_column_2pt5km: Same as `patch_start_row_2pt5km`, but for
        column instead of row.

    :return: data_dict: Dictionary with the following keys.
    data_dict["predictor_matrices"]: Same as output from `data_generator`.
    data_dict["target_matrix"]: Same as output from `data_generator`.
    data_dict["init_times_unix_sec"]: length-E numpy array of forecast-
        initialization times.
    data_dict["latitude_matrix_deg_n"]: E-by-M-by-N numpy array of latitudes
        (deg north).
    data_dict["longitude_matrix_deg_e"]: E-by-M-by-N numpy array of longitudes
        (deg east).
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
    patch_size_2pt5km_pixels = option_dict[PATCH_SIZE_KEY]

    do_residual_prediction = option_dict[DO_RESIDUAL_PREDICTION_KEY]
    predict_dewpoint_depression = option_dict[PREDICT_DEWPOINT_DEPRESSION_KEY]
    predict_gust_factor = option_dict[PREDICT_GUST_FACTOR_KEY]
    resid_baseline_model_name = option_dict[RESID_BASELINE_MODEL_KEY]
    resid_baseline_model_dir_name = option_dict[RESID_BASELINE_MODEL_DIR_KEY]
    resid_baseline_lead_time_hours = option_dict[RESID_BASELINE_LEAD_TIME_KEY]

    nwp_model_names = list(nwp_model_to_dir_name.keys())
    nwp_model_names.sort()

    # nwp_model_names = [
    #     m for m in nwp_model_names if m != nwp_model_utils.WRF_ARW_MODEL_NAME
    # ]

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

    # Do actual stuff.
    init_times_unix_sec = _find_relevant_init_times(
        first_time_by_period_unix_sec=first_init_times_unix_sec,
        last_time_by_period_unix_sec=last_init_times_unix_sec,
        nwp_model_names=nwp_model_names
    )
    num_examples = len(init_times_unix_sec)

    if nbm_constant_file_name is None:
        full_nbm_constant_matrix = None
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

        full_nbm_constant_matrix = (
            nbmct[nbm_constant_utils.DATA_KEY].values[..., field_indices]
        )

    if patch_size_2pt5km_pixels is None:
        patch_location_dict = None
    else:
        patch_location_dict = misc_utils.determine_patch_locations(
            patch_size_2pt5km_pixels=patch_size_2pt5km_pixels,
            start_row_2pt5km=patch_start_row_2pt5km,
            start_column_2pt5km=patch_start_column_2pt5km
        )

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
        patch_location_dict=patch_location_dict,
        do_residual_prediction=do_residual_prediction
    )

    if full_nbm_constant_matrix is None:
        nbm_constant_matrix = None
    else:
        these_dims = (
            target_matrix.shape[:-1] + (len(nbm_constant_field_names),)
        )
        nbm_constant_matrix = numpy.full(these_dims, numpy.nan)

    full_latitude_matrix_deg_n, full_longitude_matrix_deg_e = (
        nbm_utils.read_coords()
    )
    latitude_matrix_deg_n = numpy.full(target_matrix.shape[:-1], numpy.nan)
    longitude_matrix_deg_e = numpy.full(target_matrix.shape[:-1], numpy.nan)

    good_example_flags = numpy.full(num_examples, True, dtype=bool)

    for i in range(num_examples):
        if patch_size_2pt5km_pixels is None:
            patch_location_dict = None
        else:
            patch_location_dict = misc_utils.determine_patch_locations(
                patch_size_2pt5km_pixels=patch_size_2pt5km_pixels,
                start_row_2pt5km=patch_start_row_2pt5km,
                start_column_2pt5km=patch_start_column_2pt5km
            )

        try:
            this_target_matrix = _read_targets_one_example(
                init_time_unix_sec=init_times_unix_sec[i],
                target_lead_time_hours=target_lead_time_hours,
                target_field_names=target_field_names,
                target_dir_name=target_dir_name,
                target_norm_param_table_xarray=target_norm_param_table_xarray,
                use_quantile_norm=targets_use_quantile_norm,
                patch_location_dict=patch_location_dict
            )
        except:
            warning_string = (
                'POTENTIAL ERROR: Could not read targets for init time {0:s}.  '
                'Something went wrong in `_read_targets_one_example`.'
            ).format(
                time_conversion.unix_sec_to_string(
                    init_times_unix_sec[i], '%Y-%m-%d-%H'
                )
            )

            warnings.warn(warning_string)
            this_target_matrix = None

        if this_target_matrix is None:
            good_example_flags[i] = False
            continue

        target_matrix[i, ...] = this_target_matrix
        pld = patch_location_dict

        if pld is None:
            latitude_matrix_deg_n[i, ...] = full_latitude_matrix_deg_n
            longitude_matrix_deg_e[i, ...] = full_longitude_matrix_deg_e

            if nbm_constant_matrix is not None:
                nbm_constant_matrix[i, ...] = full_nbm_constant_matrix
        else:
            j_start = pld[misc_utils.ROW_LIMITS_2PT5KM_KEY][0]
            j_end = pld[misc_utils.ROW_LIMITS_2PT5KM_KEY][1] + 1
            k_start = pld[misc_utils.COLUMN_LIMITS_2PT5KM_KEY][0]
            k_end = pld[misc_utils.COLUMN_LIMITS_2PT5KM_KEY][1] + 1

            latitude_matrix_deg_n[i, ...] = (
                full_latitude_matrix_deg_n[j_start:j_end, k_start:k_end]
            )
            longitude_matrix_deg_e[i, ...] = (
                full_longitude_matrix_deg_e[j_start:j_end, k_start:k_end]
            )

            if nbm_constant_matrix is not None:
                nbm_constant_matrix[i, ...] = (
                    full_nbm_constant_matrix[j_start:j_end, k_start:k_end]
                )

        if do_residual_prediction:
            try:
                this_baseline_matrix = _read_residual_baseline_one_example(
                    init_time_unix_sec=init_times_unix_sec[i],
                    nwp_model_name=resid_baseline_model_name,
                    nwp_lead_time_hours=resid_baseline_lead_time_hours,
                    nwp_directory_name=resid_baseline_model_dir_name,
                    target_field_names=target_field_names,
                    patch_location_dict=patch_location_dict,
                    predict_dewpoint_depression=predict_dewpoint_depression,
                    predict_gust_factor=predict_gust_factor
                )
            except:
                warning_string = (
                    'POTENTIAL ERROR: Could not read residual baseline for '
                    'init time {0:s}.  Something went wrong in '
                    '`_read_residual_baseline_one_example`.'
                ).format(
                    time_conversion.unix_sec_to_string(
                        init_times_unix_sec[i], '%Y-%m-%d-%H'
                    )
                )

                warnings.warn(warning_string)
                this_baseline_matrix = None

            if this_baseline_matrix is None:
                good_example_flags[i] = False
                continue

            predictor_matrix_resid_baseline[i, ...] = this_baseline_matrix

        try:
            (
                this_predictor_matrix_2pt5km,
                this_predictor_matrix_10km,
                this_predictor_matrix_20km,
                this_predictor_matrix_40km,
                good_example_flags[i]
            ) = _read_predictors_one_example(
                init_time_unix_sec=init_times_unix_sec[i],
                nwp_model_names=nwp_model_names,
                nwp_lead_times_hours=nwp_lead_times_hours,
                nwp_model_to_field_names=nwp_model_to_field_names,
                nwp_model_to_dir_name=nwp_model_to_dir_name,
                nwp_norm_param_table_xarray=nwp_norm_param_table_xarray,
                use_quantile_norm=nwp_use_quantile_norm,
                patch_location_dict=patch_location_dict
            )
        except:
            warning_string = (
                'POTENTIAL ERROR: Could not read predictors for init time '
                '{0:s}.  Something went wrong in '
                '`_read_predictors_one_example`.'
            ).format(
                time_conversion.unix_sec_to_string(
                    init_times_unix_sec[i], '%Y-%m-%d-%H'
                )
            )

            warnings.warn(warning_string)
            this_predictor_matrix_2pt5km = None
            this_predictor_matrix_10km = None
            this_predictor_matrix_20km = None
            this_predictor_matrix_40km = None
            good_example_flags[i] = False

        if predictor_matrix_2pt5km is not None:
            predictor_matrix_2pt5km[i, ...] = this_predictor_matrix_2pt5km
        if predictor_matrix_10km is not None:
            predictor_matrix_10km[i, ...] = this_predictor_matrix_10km
        if predictor_matrix_20km is not None:
            predictor_matrix_20km[i, ...] = this_predictor_matrix_20km
        if predictor_matrix_40km is not None:
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
        nbm_constant_matrix = nbm_constant_matrix[good_indices, ...]
        print('Shape of NBM-constant matrix: {0:s}'.format(
            str(nbm_constant_matrix.shape)
        ))

    latitude_matrix_deg_n = latitude_matrix_deg_n[good_indices, ...]
    longitude_matrix_deg_e = longitude_matrix_deg_e[good_indices, ...]

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
        INIT_TIMES_KEY: init_times_unix_sec,
        LATITUDE_MATRIX_KEY: latitude_matrix_deg_n,
        LONGITUDE_MATRIX_KEY: longitude_matrix_deg_e
    }


def data_generator_fast_patches(option_dict, patch_overlap_size_2pt5km_pixels):
    """Fast data-generator for patchwise training.

    :param option_dict: See documentation for `data_generator`.
    :param patch_overlap_size_2pt5km_pixels: Overlap between adjacent patches,
        measured in number of pixels on the finest-resolution (2.5-km) grid.
    :return: predictor_matrices: See documentation for `data_generator`.
    :return: target_matrix: Same.
    """

    # TODO(thunderhoser): Need a create_data_fast_patches, too.

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
    patch_size_2pt5km_pixels = option_dict[PATCH_SIZE_KEY]

    do_residual_prediction = option_dict[DO_RESIDUAL_PREDICTION_KEY]
    predict_dewpoint_depression = option_dict[PREDICT_DEWPOINT_DEPRESSION_KEY]
    predict_gust_factor = option_dict[PREDICT_GUST_FACTOR_KEY]
    resid_baseline_model_name = option_dict[RESID_BASELINE_MODEL_KEY]
    resid_baseline_model_dir_name = option_dict[RESID_BASELINE_MODEL_DIR_KEY]
    resid_baseline_lead_time_hours = option_dict[RESID_BASELINE_LEAD_TIME_KEY]

    error_checking.assert_is_integer(patch_overlap_size_2pt5km_pixels)
    error_checking.assert_is_geq(patch_overlap_size_2pt5km_pixels, 16)

    nwp_model_names = list(nwp_model_to_dir_name.keys())
    nwp_model_names.sort()

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

    init_times_unix_sec = _find_relevant_init_times(
        first_time_by_period_unix_sec=first_init_times_unix_sec,
        last_time_by_period_unix_sec=last_init_times_unix_sec,
        nwp_model_names=nwp_model_names
    )

    # Do actual stuff.
    if nbm_constant_file_name is None:
        full_nbm_constant_matrix = None
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

        full_nbm_constant_matrix = (
            nbmct[nbm_constant_utils.DATA_KEY].values[..., field_indices]
        )

    numpy.random.shuffle(init_times_unix_sec)
    init_time_index = 0

    patch_metalocation_dict = {
        NUM_FULL_ROWS_KEY: len(nbm_utils.NBM_Y_COORDS_METRES),
        NUM_FULL_COLUMNS_KEY: len(nbm_utils.NBM_X_COORDS_METRES),
        NUM_PATCH_ROWS_KEY: patch_size_2pt5km_pixels,
        NUM_PATCH_COLUMNS_KEY: patch_size_2pt5km_pixels,
        PATCH_OVERLAP_SIZE_KEY: patch_overlap_size_2pt5km_pixels,
        PATCH_START_ROW_KEY: -1,
        PATCH_START_COLUMN_KEY: -1
    }

    full_target_matrix = None
    full_baseline_matrix = None
    full_predictor_matrix_2pt5km = None
    full_predictor_matrix_10km = None
    full_predictor_matrix_20km = None
    full_predictor_matrix_40km = None

    while True:
        dummy_patch_location_dict = misc_utils.determine_patch_locations(
            patch_size_2pt5km_pixels=patch_size_2pt5km_pixels
        )

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
            patch_location_dict=dummy_patch_location_dict,
            do_residual_prediction=do_residual_prediction
        )

        if full_nbm_constant_matrix is None:
            nbm_constant_matrix = None
        else:
            these_dims = (
                target_matrix.shape[:-1] + (len(nbm_constant_field_names),)
            )
            nbm_constant_matrix = numpy.full(these_dims, numpy.nan)

        num_examples_in_memory = 0

        while num_examples_in_memory < num_examples_per_batch:
            patch_metalocation_dict = __update_patch_metalocation_dict(
                patch_metalocation_dict
            )

            if patch_metalocation_dict[PATCH_START_ROW_KEY] < 0:
                full_target_matrix = None
                full_baseline_matrix = None
                full_predictor_matrix_2pt5km = None
                full_predictor_matrix_10km = None
                full_predictor_matrix_20km = None
                full_predictor_matrix_40km = None

                init_time_index, init_times_unix_sec = __increment_init_time(
                    current_index=init_time_index,
                    init_times_unix_sec=init_times_unix_sec
                )

            try:
                if full_target_matrix is None:
                    full_target_matrix = _read_targets_one_example(
                        init_time_unix_sec=init_times_unix_sec[init_time_index],
                        target_lead_time_hours=target_lead_time_hours,
                        target_field_names=target_field_names,
                        target_dir_name=target_dir_name,
                        target_norm_param_table_xarray=
                        target_norm_param_table_xarray,
                        use_quantile_norm=targets_use_quantile_norm,
                        patch_location_dict=None
                    )
            except:
                warning_string = (
                    'POTENTIAL ERROR: Could not read targets for init time '
                    '{0:s}.  Something went wrong in '
                    '`_read_targets_one_example`.'
                ).format(
                    time_conversion.unix_sec_to_string(
                        init_times_unix_sec[init_time_index], '%Y-%m-%d-%H'
                    )
                )

                warnings.warn(warning_string)
                full_target_matrix = None

            if full_target_matrix is None:
                init_time_index, init_times_unix_sec = __increment_init_time(
                    current_index=init_time_index,
                    init_times_unix_sec=init_times_unix_sec
                )
                continue

            try:
                if do_residual_prediction and full_baseline_matrix is None:
                    full_baseline_matrix = _read_residual_baseline_one_example(
                        init_time_unix_sec=init_times_unix_sec[init_time_index],
                        nwp_model_name=resid_baseline_model_name,
                        nwp_lead_time_hours=resid_baseline_lead_time_hours,
                        nwp_directory_name=resid_baseline_model_dir_name,
                        target_field_names=target_field_names,
                        patch_location_dict=None,
                        predict_dewpoint_depression=predict_dewpoint_depression,
                        predict_gust_factor=predict_gust_factor
                    )
            except:
                warning_string = (
                    'POTENTIAL ERROR: Could not read residual baseline for '
                    'init time {0:s}.  Something went wrong in '
                    '`_read_residual_baseline_one_example`.'
                ).format(
                    time_conversion.unix_sec_to_string(
                        init_times_unix_sec[init_time_index], '%Y-%m-%d-%H'
                    )
                )

                warnings.warn(warning_string)
                full_baseline_matrix = None

            if full_baseline_matrix is None:
                init_time_index, init_times_unix_sec = __increment_init_time(
                    current_index=init_time_index,
                    init_times_unix_sec=init_times_unix_sec
                )
                continue

            try:
                if full_predictor_matrix_2pt5km is None:
                    (
                        full_predictor_matrix_2pt5km,
                        full_predictor_matrix_10km,
                        full_predictor_matrix_20km,
                        full_predictor_matrix_40km,
                        found_any_predictors
                    ) = _read_predictors_one_example(
                        init_time_unix_sec=init_times_unix_sec[init_time_index],
                        nwp_model_names=nwp_model_names,
                        nwp_lead_times_hours=nwp_lead_times_hours,
                        nwp_model_to_field_names=nwp_model_to_field_names,
                        nwp_model_to_dir_name=nwp_model_to_dir_name,
                        nwp_norm_param_table_xarray=nwp_norm_param_table_xarray,
                        use_quantile_norm=nwp_use_quantile_norm,
                        patch_location_dict=None
                    )
                else:
                    found_any_predictors = False
            except:
                warning_string = (
                    'POTENTIAL ERROR: Could not read predictors for init time '
                    '{0:s}.  Something went wrong in '
                    '`_read_predictors_one_example`.'
                ).format(
                    time_conversion.unix_sec_to_string(
                        init_times_unix_sec[init_time_index], '%Y-%m-%d-%H'
                    )
                )

                warnings.warn(warning_string)
                full_predictor_matrix_2pt5km = None
                full_predictor_matrix_10km = None
                full_predictor_matrix_20km = None
                full_predictor_matrix_40km = None
                found_any_predictors = False

            if not found_any_predictors:
                init_time_index, init_times_unix_sec = __increment_init_time(
                    current_index=init_time_index,
                    init_times_unix_sec=init_times_unix_sec
                )
                continue

            patch_location_dict = misc_utils.determine_patch_locations(
                patch_size_2pt5km_pixels=patch_size_2pt5km_pixels,
                start_row_2pt5km=patch_metalocation_dict[PATCH_START_ROW_KEY],
                start_column_2pt5km=
                patch_metalocation_dict[PATCH_START_COLUMN_KEY]
            )
            print(patch_location_dict)
            pld = patch_location_dict

            j_start = pld[misc_utils.ROW_LIMITS_2PT5KM_KEY][0]
            j_end = pld[misc_utils.ROW_LIMITS_2PT5KM_KEY][1] + 1
            k_start = pld[misc_utils.COLUMN_LIMITS_2PT5KM_KEY][0]
            k_end = pld[misc_utils.COLUMN_LIMITS_2PT5KM_KEY][1] + 1
            i = num_examples_in_memory + 0

            target_matrix[i, ...] = (
                full_target_matrix[j_start:j_end, k_start:k_end, ...]
            )

            if do_residual_prediction:
                predictor_matrix_resid_baseline[i, ...] = (
                    full_baseline_matrix[j_start:j_end, k_start:k_end, ...]
                )

            if predictor_matrix_2pt5km is not None:
                predictor_matrix_2pt5km[i, ...] = (
                    full_predictor_matrix_2pt5km[j_start:j_end, k_start:k_end, ...]
                )

            if nbm_constant_matrix is not None:
                nbm_constant_matrix[i, ...] = (
                    full_nbm_constant_matrix[j_start:j_end, k_start:k_end, ...]
                )

            if predictor_matrix_10km is not None:
                j_start = pld[misc_utils.ROW_LIMITS_10KM_KEY][0]
                j_end = pld[misc_utils.ROW_LIMITS_10KM_KEY][1] + 1
                k_start = pld[misc_utils.COLUMN_LIMITS_10KM_KEY][0]
                k_end = pld[misc_utils.COLUMN_LIMITS_10KM_KEY][1] + 1

                predictor_matrix_10km[i, ...] = (
                    full_predictor_matrix_10km[j_start:j_end, k_start:k_end, ...]
                )

            if predictor_matrix_20km is not None:
                j_start = pld[misc_utils.ROW_LIMITS_20KM_KEY][0]
                j_end = pld[misc_utils.ROW_LIMITS_20KM_KEY][1] + 1
                k_start = pld[misc_utils.COLUMN_LIMITS_20KM_KEY][0]
                k_end = pld[misc_utils.COLUMN_LIMITS_20KM_KEY][1] + 1

                predictor_matrix_20km[i, ...] = (
                    full_predictor_matrix_20km[j_start:j_end, k_start:k_end, ...]
                )

            if predictor_matrix_40km is not None:
                j_start = pld[misc_utils.ROW_LIMITS_40KM_KEY][0]
                j_end = pld[misc_utils.ROW_LIMITS_40KM_KEY][1] + 1
                k_start = pld[misc_utils.COLUMN_LIMITS_40KM_KEY][0]
                k_end = pld[misc_utils.COLUMN_LIMITS_40KM_KEY][1] + 1

                predictor_matrix_40km[i, ...] = (
                    full_predictor_matrix_40km[j_start:j_end, k_start:k_end, ...]
                )

            num_examples_in_memory += 1

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
    option_dict["patch_size_2pt5km_pixels"]: Patch size, in units of 2.5-km
        pixels.  For example, if patch_size_2pt5km_pixels = 448, then grid
        dimensions at the finest resolution (2.5 km) are 448 x 448.  If you
        want to train with the full grid -- and not the patchwise approach --
        make this argument None.
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
    patch_size_2pt5km_pixels = option_dict[PATCH_SIZE_KEY]

    do_residual_prediction = option_dict[DO_RESIDUAL_PREDICTION_KEY]
    predict_dewpoint_depression = option_dict[PREDICT_DEWPOINT_DEPRESSION_KEY]
    predict_gust_factor = option_dict[PREDICT_GUST_FACTOR_KEY]
    resid_baseline_model_name = option_dict[RESID_BASELINE_MODEL_KEY]
    resid_baseline_model_dir_name = option_dict[RESID_BASELINE_MODEL_DIR_KEY]
    resid_baseline_lead_time_hours = option_dict[RESID_BASELINE_LEAD_TIME_KEY]

    nwp_model_names = list(nwp_model_to_dir_name.keys())
    nwp_model_names.sort()

    # nwp_model_names = [
    #     m for m in nwp_model_names if m != nwp_model_utils.WRF_ARW_MODEL_NAME
    # ]

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

    init_times_unix_sec = _find_relevant_init_times(
        first_time_by_period_unix_sec=first_init_times_unix_sec,
        last_time_by_period_unix_sec=last_init_times_unix_sec,
        nwp_model_names=nwp_model_names
    )

    # Do actual stuff.
    if nbm_constant_file_name is None:
        full_nbm_constant_matrix = None
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

        full_nbm_constant_matrix = (
            nbmct[nbm_constant_utils.DATA_KEY].values[..., field_indices]
        )

    init_time_index = len(init_times_unix_sec)

    while True:
        if patch_size_2pt5km_pixels is None:
            patch_location_dict = None
        else:
            patch_location_dict = misc_utils.determine_patch_locations(
                patch_size_2pt5km_pixels=patch_size_2pt5km_pixels
            )

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
            patch_location_dict=patch_location_dict,
            do_residual_prediction=do_residual_prediction
        )

        if full_nbm_constant_matrix is None:
            nbm_constant_matrix = None
        else:
            these_dims = (
                target_matrix.shape[:-1] + (len(nbm_constant_field_names),)
            )
            nbm_constant_matrix = numpy.full(these_dims, numpy.nan)

        num_examples_in_memory = 0

        while num_examples_in_memory < num_examples_per_batch:
            if init_time_index == len(init_times_unix_sec):
                numpy.random.shuffle(init_times_unix_sec)
                init_time_index = 0

            if patch_size_2pt5km_pixels is None:
                patch_location_dict = None
            else:
                patch_location_dict = misc_utils.determine_patch_locations(
                    patch_size_2pt5km_pixels=patch_size_2pt5km_pixels
                )

            try:
                this_target_matrix = _read_targets_one_example(
                    init_time_unix_sec=init_times_unix_sec[init_time_index],
                    target_lead_time_hours=target_lead_time_hours,
                    target_field_names=target_field_names,
                    target_dir_name=target_dir_name,
                    target_norm_param_table_xarray=target_norm_param_table_xarray,
                    use_quantile_norm=targets_use_quantile_norm,
                    patch_location_dict=patch_location_dict
                )
            except:
                warning_string = (
                    'POTENTIAL ERROR: Could not read targets for init time '
                    '{0:s}.  Something went wrong in '
                    '`_read_targets_one_example`.'
                ).format(
                    time_conversion.unix_sec_to_string(
                        init_times_unix_sec[init_time_index], '%Y-%m-%d-%H'
                    )
                )

                warnings.warn(warning_string)
                this_target_matrix = None

            if this_target_matrix is None:
                init_time_index += 1
                continue

            i = num_examples_in_memory + 0
            target_matrix[i, ...] = this_target_matrix

            if nbm_constant_matrix is not None:
                pld = patch_location_dict

                if pld is None:
                    nbm_constant_matrix[i, ...] = full_nbm_constant_matrix
                else:
                    j_start = pld[misc_utils.ROW_LIMITS_2PT5KM_KEY][0]
                    j_end = pld[misc_utils.ROW_LIMITS_2PT5KM_KEY][1] + 1
                    k_start = pld[misc_utils.COLUMN_LIMITS_2PT5KM_KEY][0]
                    k_end = pld[misc_utils.COLUMN_LIMITS_2PT5KM_KEY][1] + 1

                    nbm_constant_matrix[i, ...] = (
                        full_nbm_constant_matrix[j_start:j_end, k_start:k_end]
                    )

            if do_residual_prediction:
                try:
                    this_baseline_matrix = _read_residual_baseline_one_example(
                        init_time_unix_sec=init_times_unix_sec[init_time_index],
                        nwp_model_name=resid_baseline_model_name,
                        nwp_lead_time_hours=resid_baseline_lead_time_hours,
                        nwp_directory_name=resid_baseline_model_dir_name,
                        target_field_names=target_field_names,
                        patch_location_dict=patch_location_dict,
                        predict_dewpoint_depression=predict_dewpoint_depression,
                        predict_gust_factor=predict_gust_factor
                    )
                except:
                    warning_string = (
                        'POTENTIAL ERROR: Could not read residual baseline for '
                        'init time {0:s}.  Something went wrong in '
                        '`_read_residual_baseline_one_example`.'
                    ).format(
                        time_conversion.unix_sec_to_string(
                            init_times_unix_sec[init_time_index], '%Y-%m-%d-%H'
                        )
                    )

                    warnings.warn(warning_string)
                    this_baseline_matrix = None

                if this_baseline_matrix is None:
                    init_time_index += 1
                    continue

                predictor_matrix_resid_baseline[i, ...] = this_baseline_matrix

            try:
                (
                    this_predictor_matrix_2pt5km,
                    this_predictor_matrix_10km,
                    this_predictor_matrix_20km,
                    this_predictor_matrix_40km,
                    found_any_predictors
                ) = _read_predictors_one_example(
                    init_time_unix_sec=init_times_unix_sec[init_time_index],
                    nwp_model_names=nwp_model_names,
                    nwp_lead_times_hours=nwp_lead_times_hours,
                    nwp_model_to_field_names=nwp_model_to_field_names,
                    nwp_model_to_dir_name=nwp_model_to_dir_name,
                    nwp_norm_param_table_xarray=nwp_norm_param_table_xarray,
                    use_quantile_norm=nwp_use_quantile_norm,
                    patch_location_dict=patch_location_dict
                )
            except:
                warning_string = (
                    'POTENTIAL ERROR: Could not read predictors for init time '
                    '{0:s}.  Something went wrong in '
                    '`_read_predictors_one_example`.'
                ).format(
                    time_conversion.unix_sec_to_string(
                        init_times_unix_sec[init_time_index], '%Y-%m-%d-%H'
                    )
                )

                warnings.warn(warning_string)
                this_predictor_matrix_2pt5km = None
                this_predictor_matrix_10km = None
                this_predictor_matrix_20km = None
                this_predictor_matrix_40km = None
                found_any_predictors = False

            if not found_any_predictors:
                init_time_index += 1
                continue

            if predictor_matrix_2pt5km is not None:
                predictor_matrix_2pt5km[i, ...] = this_predictor_matrix_2pt5km
            if predictor_matrix_10km is not None:
                predictor_matrix_10km[i, ...] = this_predictor_matrix_10km
            if predictor_matrix_20km is not None:
                predictor_matrix_20km[i, ...] = this_predictor_matrix_20km
            if predictor_matrix_40km is not None:
                predictor_matrix_40km[i, ...] = this_predictor_matrix_40km

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
        patch_overlap_fast_gen_2pt5km_pixels, output_dir_name):
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
    :param patch_overlap_fast_gen_2pt5km_pixels: See documentation for
        `data_generator_fast_patches`.
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

    if patch_overlap_fast_gen_2pt5km_pixels is None:
        training_generator = data_generator(training_option_dict)
        validation_generator = data_generator(validation_option_dict)
    else:
        training_generator = data_generator_fast_patches(
            option_dict=training_option_dict,
            patch_overlap_size_2pt5km_pixels=
            patch_overlap_fast_gen_2pt5km_pixels
        )
        validation_generator = data_generator_fast_patches(
            option_dict=validation_option_dict,
            patch_overlap_size_2pt5km_pixels=
            patch_overlap_fast_gen_2pt5km_pixels
        )

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
        early_stopping_patience_epochs=early_stopping_patience_epochs,
        patch_overlap_fast_gen_2pt5km_pixels=
        patch_overlap_fast_gen_2pt5km_pixels
    )

    model_object.fit(
        x=training_generator,
        steps_per_epoch=num_training_batches_per_epoch,
        epochs=num_epochs, verbose=1, callbacks=list_of_callback_objects,
        validation_data=validation_generator,
        validation_steps=num_validation_batches_per_epoch
    )


def apply_patchwise_model_to_full_grid(
        model_object, full_predictor_matrices, num_examples_per_batch,
        predict_dewpoint_depression, predict_gust_factor,
        patch_overlap_size_2pt5km_pixels,
        verbose=True, target_field_names=None):
    """Does inference for neural net trained with patchwise approach.

    :param model_object: See documentation for `apply_model`.
    :param full_predictor_matrices: See documentation for `apply_model` --
        except these matrices are on the full grid.
    :param num_examples_per_batch: Same.
    :param predict_dewpoint_depression: Same.
    :param predict_gust_factor: Same.
    :param patch_overlap_size_2pt5km_pixels: Overlap between adjacent patches,
        measured in number of pixels on the finest-resolution (2.5-km) grid.
    :param verbose: See documentation for `apply_model`.
    :param target_field_names: Same.
    :return: full_prediction_matrix: See documentation for `apply_model` --
        except this matrix is on the full grid.
    """

    # Check input args.
    these_dim = model_object.layers[-1].output.shape
    num_rows_in_patch = these_dim[1]
    num_columns_in_patch = these_dim[2]
    num_target_fields = these_dim[3]

    # TODO(thunderhoser): Might relax this constraint eventually -- I don't
    # know.
    error_checking.assert_equals(num_rows_in_patch, num_columns_in_patch)

    error_checking.assert_is_integer(patch_overlap_size_2pt5km_pixels)
    error_checking.assert_is_geq(patch_overlap_size_2pt5km_pixels, 0)
    error_checking.assert_is_less_than(
        2 * patch_overlap_size_2pt5km_pixels,
        min([num_rows_in_patch, num_columns_in_patch])
    )

    error_checking.assert_is_boolean(verbose)

    # Do actual stuff.
    num_rows_2pt5km = len(nbm_utils.NBM_Y_COORDS_METRES)
    num_columns_2pt5km = len(nbm_utils.NBM_X_COORDS_METRES)

    patch_metalocation_dict = {
        NUM_FULL_ROWS_KEY: num_rows_2pt5km,
        NUM_FULL_COLUMNS_KEY: num_columns_2pt5km,
        NUM_PATCH_ROWS_KEY: num_rows_in_patch,
        NUM_PATCH_COLUMNS_KEY: num_columns_in_patch,
        PATCH_OVERLAP_SIZE_KEY: patch_overlap_size_2pt5km_pixels,
        PATCH_START_ROW_KEY: -1,
        PATCH_START_COLUMN_KEY: -1
    }

    num_examples = full_predictor_matrices[0].shape[0]
    these_dim = (
        num_examples, num_rows_2pt5km, num_columns_2pt5km, num_target_fields
    )
    summed_prediction_matrix = numpy.full(these_dim, 0.)
    prediction_count_matrix = numpy.full(these_dim, 0, dtype=int)

    while True:
        patch_metalocation_dict = __update_patch_metalocation_dict(
            patch_metalocation_dict
        )

        patch_start_row_2pt5km = patch_metalocation_dict[PATCH_START_ROW_KEY]
        if patch_start_row_2pt5km < 0:
            break

        patch_location_dict = misc_utils.determine_patch_locations(
            patch_size_2pt5km_pixels=num_rows_in_patch,
            start_row_2pt5km=patch_metalocation_dict[PATCH_START_ROW_KEY],
            start_column_2pt5km=patch_metalocation_dict[PATCH_START_COLUMN_KEY]
        )
        pld = patch_location_dict

        # TODO(thunderhoser): Allow for inner and outer domains -- but not now!
        if verbose:
            i_start = pld[misc_utils.ROW_LIMITS_2PT5KM_KEY][0]
            i_end = pld[misc_utils.ROW_LIMITS_2PT5KM_KEY][1]
            j_start = pld[misc_utils.COLUMN_LIMITS_2PT5KM_KEY][0]
            j_end = pld[misc_utils.COLUMN_LIMITS_2PT5KM_KEY][1]

            print((
                'Applying model to rows {0:d}-{1:d} of {2:d}, and '
                'columns {3:d}-{4:d} of {5:d}, in finest-resolution grid...'
            ).format(
                i_start, i_end, num_rows_2pt5km,
                j_start, j_end, num_columns_2pt5km
            ))

        patch_predictor_matrices = []

        for this_full_pred_matrix in full_predictor_matrices:
            this_downsampling_factor = int(numpy.round(
                float(num_rows_2pt5km) /
                this_full_pred_matrix.shape[1]
            ))
            assert this_downsampling_factor in POSSIBLE_DOWNSAMPLING_FACTORS

            if this_downsampling_factor == 1:
                i_start = pld[misc_utils.ROW_LIMITS_2PT5KM_KEY][0]
                i_end = pld[misc_utils.ROW_LIMITS_2PT5KM_KEY][1] + 1
                j_start = pld[misc_utils.COLUMN_LIMITS_2PT5KM_KEY][0]
                j_end = pld[misc_utils.COLUMN_LIMITS_2PT5KM_KEY][1] + 1
            elif this_downsampling_factor == 4:
                i_start = pld[misc_utils.ROW_LIMITS_10KM_KEY][0]
                i_end = pld[misc_utils.ROW_LIMITS_10KM_KEY][1] + 1
                j_start = pld[misc_utils.COLUMN_LIMITS_10KM_KEY][0]
                j_end = pld[misc_utils.COLUMN_LIMITS_10KM_KEY][1] + 1
            elif this_downsampling_factor == 8:
                i_start = pld[misc_utils.ROW_LIMITS_20KM_KEY][0]
                i_end = pld[misc_utils.ROW_LIMITS_20KM_KEY][1] + 1
                j_start = pld[misc_utils.COLUMN_LIMITS_20KM_KEY][0]
                j_end = pld[misc_utils.COLUMN_LIMITS_20KM_KEY][1] + 1
            else:
                i_start = pld[misc_utils.ROW_LIMITS_40KM_KEY][0]
                i_end = pld[misc_utils.ROW_LIMITS_40KM_KEY][1] + 1
                j_start = pld[misc_utils.COLUMN_LIMITS_40KM_KEY][0]
                j_end = pld[misc_utils.COLUMN_LIMITS_40KM_KEY][1] + 1

            patch_predictor_matrices.append(
                this_full_pred_matrix[:, i_start:i_end, j_start:j_end, ...]
            )

        i_start = pld[misc_utils.ROW_LIMITS_2PT5KM_KEY][0]
        i_end = pld[misc_utils.ROW_LIMITS_2PT5KM_KEY][1] + 1
        j_start = pld[misc_utils.COLUMN_LIMITS_2PT5KM_KEY][0]
        j_end = pld[misc_utils.COLUMN_LIMITS_2PT5KM_KEY][1] + 1

        patch_prediction_matrix = apply_model(
            model_object=model_object,
            predictor_matrices=patch_predictor_matrices,
            num_examples_per_batch=num_examples_per_batch,
            predict_dewpoint_depression=predict_dewpoint_depression,
            predict_gust_factor=predict_gust_factor,
            verbose=False,
            target_field_names=target_field_names
        )

        summed_prediction_matrix[:, i_start:i_end, j_start:j_end, :] += (
            patch_prediction_matrix
        )
        prediction_count_matrix[:, i_start:i_end, j_start:j_end, :] += 1

    if verbose:
        print('Have applied model everywhere in full grid!')

    prediction_count_matrix = prediction_count_matrix.astype(float)
    prediction_count_matrix[prediction_count_matrix < 0.01] = numpy.nan
    return summed_prediction_matrix / prediction_count_matrix


def apply_patchwise_model_to_full_grid_test(
        full_predictor_matrices,
        patch_overlap_size_2pt5km_pixels,
        verbose=True):
    """Does inference for neural net trained with patchwise approach.

    :param model_object: See documentation for `apply_model`.
    :param full_predictor_matrices: See documentation for `apply_model` --
        except these matrices are on the full grid.
    :param num_examples_per_batch: Same.
    :param predict_dewpoint_depression: Same.
    :param predict_gust_factor: Same.
    :param patch_overlap_size_2pt5km_pixels: Overlap between adjacent patches,
        measured in number of pixels on the finest-resolution (2.5-km) grid.
    :param verbose: See documentation for `apply_model`.
    :param target_field_names: Same.
    :return: full_prediction_matrix: See documentation for `apply_model` --
        except this matrix is on the full grid.
    """

    # Check input args.
    these_dim = (None, 448, 448, 2)
    num_rows_in_patch = these_dim[1]
    num_columns_in_patch = these_dim[2]
    num_target_fields = these_dim[3]

    # TODO(thunderhoser): Might relax this constraint eventually -- I don't
    # know.
    error_checking.assert_equals(num_rows_in_patch, num_columns_in_patch)

    error_checking.assert_is_integer(patch_overlap_size_2pt5km_pixels)
    error_checking.assert_is_geq(patch_overlap_size_2pt5km_pixels, 0)
    error_checking.assert_is_less_than(
        2 * patch_overlap_size_2pt5km_pixels,
        min([num_rows_in_patch, num_columns_in_patch])
    )

    error_checking.assert_is_boolean(verbose)

    # Do actual stuff.
    num_rows_2pt5km = len(nbm_utils.NBM_Y_COORDS_METRES)
    num_columns_2pt5km = len(nbm_utils.NBM_X_COORDS_METRES)

    patch_metalocation_dict = {
        NUM_FULL_ROWS_KEY: num_rows_2pt5km,
        NUM_FULL_COLUMNS_KEY: num_columns_2pt5km,
        NUM_PATCH_ROWS_KEY: num_rows_in_patch,
        NUM_PATCH_COLUMNS_KEY: num_columns_in_patch,
        PATCH_OVERLAP_SIZE_KEY: patch_overlap_size_2pt5km_pixels,
        PATCH_START_ROW_KEY: -1,
        PATCH_START_COLUMN_KEY: -1
    }

    num_examples = full_predictor_matrices[0].shape[0]
    these_dim = (
        num_examples, num_rows_2pt5km, num_columns_2pt5km, num_target_fields
    )
    summed_prediction_matrix = numpy.full(these_dim, 0.)
    prediction_count_matrix = numpy.full(these_dim, 0, dtype=int)

    while True:
        patch_metalocation_dict = __update_patch_metalocation_dict(
            patch_metalocation_dict
        )

        patch_start_row_2pt5km = patch_metalocation_dict[PATCH_START_ROW_KEY]
        if patch_start_row_2pt5km < 0:
            break

        patch_location_dict = misc_utils.determine_patch_locations(
            patch_size_2pt5km_pixels=num_rows_in_patch,
            start_row_2pt5km=patch_metalocation_dict[PATCH_START_ROW_KEY],
            start_column_2pt5km=patch_metalocation_dict[PATCH_START_COLUMN_KEY]
        )
        pld = patch_location_dict

        # TODO(thunderhoser): Allow for inner and outer domains -- but not now!
        if verbose:
            i_start = pld[misc_utils.ROW_LIMITS_2PT5KM_KEY][0]
            i_end = pld[misc_utils.ROW_LIMITS_2PT5KM_KEY][1]
            j_start = pld[misc_utils.COLUMN_LIMITS_2PT5KM_KEY][0]
            j_end = pld[misc_utils.COLUMN_LIMITS_2PT5KM_KEY][1]

            print((
                'Applying model to rows {0:d}-{1:d} of {2:d}, and '
                'columns {3:d}-{4:d} of {5:d}, in finest-resolution grid...'
            ).format(
                i_start, i_end, num_rows_2pt5km,
                j_start, j_end, num_columns_2pt5km
            ))

        patch_predictor_matrices = []

        for this_full_pred_matrix in full_predictor_matrices:
            this_downsampling_factor = int(numpy.round(
                float(num_rows_2pt5km) /
                this_full_pred_matrix.shape[1]
            ))
            assert this_downsampling_factor in POSSIBLE_DOWNSAMPLING_FACTORS

            if this_downsampling_factor == 1:
                i_start = pld[misc_utils.ROW_LIMITS_2PT5KM_KEY][0]
                i_end = pld[misc_utils.ROW_LIMITS_2PT5KM_KEY][1] + 1
                j_start = pld[misc_utils.COLUMN_LIMITS_2PT5KM_KEY][0]
                j_end = pld[misc_utils.COLUMN_LIMITS_2PT5KM_KEY][1] + 1
            elif this_downsampling_factor == 4:
                i_start = pld[misc_utils.ROW_LIMITS_10KM_KEY][0]
                i_end = pld[misc_utils.ROW_LIMITS_10KM_KEY][1] + 1
                j_start = pld[misc_utils.COLUMN_LIMITS_10KM_KEY][0]
                j_end = pld[misc_utils.COLUMN_LIMITS_10KM_KEY][1] + 1
            elif this_downsampling_factor == 8:
                i_start = pld[misc_utils.ROW_LIMITS_20KM_KEY][0]
                i_end = pld[misc_utils.ROW_LIMITS_20KM_KEY][1] + 1
                j_start = pld[misc_utils.COLUMN_LIMITS_20KM_KEY][0]
                j_end = pld[misc_utils.COLUMN_LIMITS_20KM_KEY][1] + 1
            else:
                i_start = pld[misc_utils.ROW_LIMITS_40KM_KEY][0]
                i_end = pld[misc_utils.ROW_LIMITS_40KM_KEY][1] + 1
                j_start = pld[misc_utils.COLUMN_LIMITS_40KM_KEY][0]
                j_end = pld[misc_utils.COLUMN_LIMITS_40KM_KEY][1] + 1

            patch_predictor_matrices.append(
                this_full_pred_matrix[:, i_start:i_end, j_start:j_end, ...]
            )

        i_start = pld[misc_utils.ROW_LIMITS_2PT5KM_KEY][0]
        i_end = pld[misc_utils.ROW_LIMITS_2PT5KM_KEY][1] + 1
        j_start = pld[misc_utils.COLUMN_LIMITS_2PT5KM_KEY][0]
        j_end = pld[misc_utils.COLUMN_LIMITS_2PT5KM_KEY][1] + 1

        # patch_prediction_matrix = apply_model(
        #     model_object=model_object,
        #     predictor_matrices=patch_predictor_matrices,
        #     num_examples_per_batch=num_examples_per_batch,
        #     predict_dewpoint_depression=predict_dewpoint_depression,
        #     predict_gust_factor=predict_gust_factor,
        #     verbose=False,
        #     target_field_names=target_field_names
        # )

        patch_prediction_matrix = numpy.random.normal(
            loc=0., scale=1., size=(num_examples, num_rows_in_patch, num_columns_in_patch, num_target_fields)
        )

        summed_prediction_matrix[:, i_start:i_end, j_start:j_end, :] += (
            patch_prediction_matrix
        )
        prediction_count_matrix[:, i_start:i_end, j_start:j_end, :] += 1

    if verbose:
        print('Have applied model everywhere in full grid!')

    prediction_count_matrix = prediction_count_matrix.astype(float)
    prediction_count_matrix[prediction_count_matrix < 0.01] = numpy.nan
    return summed_prediction_matrix / prediction_count_matrix


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
        plateau_learning_rate_multiplier, early_stopping_patience_epochs,
        patch_overlap_fast_gen_2pt5km_pixels):
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
    :param patch_overlap_fast_gen_2pt5km_pixels: Same.
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
        EARLY_STOPPING_PATIENCE_KEY: early_stopping_patience_epochs,
        PATCH_OVERLAP_FOR_FAST_GEN_KEY: patch_overlap_fast_gen_2pt5km_pixels
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
    metadata_dict["patch_overlap_fast_gen_2pt5km_pixels"]: Same.

    :raises: ValueError: if any expected key is not found in dictionary.
    """

    error_checking.assert_file_exists(pickle_file_name)

    pickle_file_handle = open(pickle_file_name, 'rb')
    metadata_dict = pickle.load(pickle_file_handle)
    pickle_file_handle.close()

    if PATCH_OVERLAP_FOR_FAST_GEN_KEY not in metadata_dict:
        metadata_dict[PATCH_OVERLAP_FOR_FAST_GEN_KEY] = None

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

"""NN training and inference with simple (not multi-patch) approach.

The simple approach can still mean one of two things:

- Training with data on the full NBM grid (not recommended, due to memory
  constraints)
- Training with data on a single patch -- i.e., one subdomain of the NBM grid,
  which does not move around
"""

import os
import warnings
import numpy
import pandas
import keras
from ml_for_national_blend.io import example_io
from ml_for_national_blend.io import nwp_model_io
from ml_for_national_blend.io import urma_io
from ml_for_national_blend.io import nbm_constant_io
from ml_for_national_blend.utils import nwp_model_utils
from ml_for_national_blend.utils import misc_utils
from ml_for_national_blend.utils import urma_utils
from ml_for_national_blend.utils import nbm_utils
from ml_for_national_blend.utils import nbm_constant_utils
from ml_for_national_blend.outside_code import time_conversion
from ml_for_national_blend.outside_code import file_system_utils
from ml_for_national_blend.outside_code import error_checking
from ml_for_national_blend.machine_learning import neural_net_utils as nn_utils
from ml_for_national_blend.machine_learning import nwp_input

PREDICTOR_MATRICES_KEY = 'predictor_matrices_key'
TARGET_MATRIX_KEY = 'target_matrix'
INIT_TIMES_KEY = 'init_times_unix_sec'
LATITUDE_MATRIX_KEY = 'latitude_matrix_deg_n'
LONGITUDE_MATRIX_KEY = 'longitude_matrix_deg_e'


def create_data(option_dict, init_time_unix_sec,
                return_predictors_as_dict=False):
    """Creates validation or testing data for neural network.

    E = number of examples (data samples) = 1
    M = number of rows in full-resolution (2.5-km) grid
    N = number of columns in full-resolution (2.5-km) grid

    :param option_dict: See documentation for `data_generator`.
    :param init_time_unix_sec: Will return data only for this initialization
        time.
    :param return_predictors_as_dict: See documentation for
        `neural_net_utils.create_data_dict_or_tuple`.
    :return: data_dict: Dictionary with the following keys.
    data_dict["predictor_matrices"]: Same as output from `data_generator`.
    data_dict["target_matrix"]: Same as output from `data_generator`.
    data_dict["init_times_unix_sec"]: length-E numpy array of forecast-
        initialization times.
    data_dict["latitude_matrix_deg_n"]: E-by-M-by-N numpy array of grid-point
        latitudes (deg north).
    data_dict["longitude_matrix_deg_e"]: E-by-M-by-N numpy array of grid-point
        longitudes (deg east).
    """

    # Add dummy arguments.
    error_checking.assert_is_integer(init_time_unix_sec)

    option_dict[nn_utils.BATCH_SIZE_KEY] = 32
    option_dict[nn_utils.FIRST_INIT_TIMES_KEY] = numpy.array(
        [init_time_unix_sec], dtype=int
    )
    option_dict[nn_utils.LAST_INIT_TIMES_KEY] = numpy.array(
        [init_time_unix_sec], dtype=int
    )

    # Check input args.
    option_dict = nn_utils.check_generator_args(option_dict)
    first_init_times_unix_sec = option_dict[nn_utils.FIRST_INIT_TIMES_KEY]
    last_init_times_unix_sec = option_dict[nn_utils.LAST_INIT_TIMES_KEY]
    nwp_lead_times_hours = option_dict[nn_utils.NWP_LEAD_TIMES_KEY]
    nwp_model_to_dir_name = option_dict[nn_utils.NWP_MODEL_TO_DIR_KEY]
    nwp_model_to_field_names = option_dict[nn_utils.NWP_MODEL_TO_FIELDS_KEY]
    nwp_normalization_file_name = option_dict[nn_utils.NWP_NORM_FILE_KEY]
    nwp_resid_norm_file_name = option_dict[nn_utils.NWP_RESID_NORM_FILE_KEY]
    nwp_use_quantile_norm = option_dict[nn_utils.NWP_USE_QUANTILE_NORM_KEY]
    backup_nwp_model_name = option_dict[nn_utils.BACKUP_NWP_MODEL_KEY]
    backup_nwp_directory_name = option_dict[nn_utils.BACKUP_NWP_DIR_KEY]
    target_lead_time_hours = option_dict[nn_utils.TARGET_LEAD_TIME_KEY]
    target_field_names = option_dict[nn_utils.TARGET_FIELDS_KEY]
    target_lag_times_hours = option_dict[nn_utils.TARGET_LAG_TIMES_KEY]
    target_dir_name = option_dict[nn_utils.TARGET_DIR_KEY]
    target_normalization_file_name = option_dict[nn_utils.TARGET_NORM_FILE_KEY]
    target_resid_norm_file_name = option_dict[
        nn_utils.TARGET_RESID_NORM_FILE_KEY
    ]
    targets_use_quantile_norm = option_dict[
        nn_utils.TARGETS_USE_QUANTILE_NORM_KEY
    ]
    recent_bias_init_time_lags_hours = option_dict[
        nn_utils.RECENT_BIAS_LAG_TIMES_KEY
    ]
    recent_bias_lead_times_hours = option_dict[
        nn_utils.RECENT_BIAS_LEAD_TIMES_KEY
    ]
    nbm_constant_field_names = option_dict[nn_utils.NBM_CONSTANT_FIELDS_KEY]
    nbm_constant_file_name = option_dict[nn_utils.NBM_CONSTANT_FILE_KEY]
    compare_to_baseline_in_loss = option_dict[
        nn_utils.COMPARE_TO_BASELINE_IN_LOSS_KEY
    ]
    sentinel_value = option_dict[nn_utils.SENTINEL_VALUE_KEY]
    patch_size_2pt5km_pixels = option_dict[nn_utils.PATCH_SIZE_KEY]
    patch_buffer_size_2pt5km_pixels = option_dict[
        nn_utils.PATCH_BUFFER_SIZE_KEY
    ]
    patch_start_row_2pt5km = option_dict[nn_utils.PATCH_START_ROW_KEY]
    patch_start_column_2pt5km = option_dict[nn_utils.PATCH_START_COLUMN_KEY]
    require_all_predictors = option_dict[nn_utils.REQUIRE_ALL_PREDICTORS_KEY]

    do_residual_prediction = option_dict[nn_utils.DO_RESIDUAL_PREDICTION_KEY]
    resid_baseline_model_name = option_dict[nn_utils.RESID_BASELINE_MODEL_KEY]
    resid_baseline_model_dir_name = option_dict[
        nn_utils.RESID_BASELINE_MODEL_DIR_KEY
    ]
    resid_baseline_lead_time_hours = option_dict[
        nn_utils.RESID_BASELINE_LEAD_TIME_KEY
    ]

    use_recent_biases = not (
        recent_bias_init_time_lags_hours is None
        or recent_bias_lead_times_hours is None
    )

    if patch_size_2pt5km_pixels is None:
        num_rows, num_columns = nwp_input.get_grid_dimensions(
            grid_spacing_km=2.5, patch_location_dict=None
        )
        mask_matrix_for_loss = numpy.full((num_rows, num_columns), 1, dtype=int)
    else:
        mask_matrix_for_loss = nn_utils.patch_buffer_to_mask(
            patch_size_2pt5km_pixels=patch_size_2pt5km_pixels,
            patch_buffer_size_2pt5km_pixels=patch_buffer_size_2pt5km_pixels
        )

    mask_matrix_for_loss = numpy.expand_dims(mask_matrix_for_loss, axis=0)
    mask_matrix_for_loss = numpy.expand_dims(mask_matrix_for_loss, axis=-1)

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

    if nwp_resid_norm_file_name is None:
        nwp_resid_norm_param_table_xarray = None
    else:
        print('Reading residual-normalization params from: "{0:s}"...'.format(
            nwp_resid_norm_file_name
        ))
        nwp_resid_norm_param_table_xarray = (
            nwp_model_io.read_normalization_file(nwp_resid_norm_file_name)
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

    if target_resid_norm_file_name is None:
        target_resid_norm_param_table_xarray = None
    else:
        print('Reading residual-normalization params from: "{0:s}"...'.format(
            target_resid_norm_file_name
        ))
        target_resid_norm_param_table_xarray = urma_io.read_normalization_file(
            target_resid_norm_file_name
        )

    init_times_unix_sec = nn_utils.find_relevant_init_times(
        first_time_by_period_unix_sec=first_init_times_unix_sec,
        last_time_by_period_unix_sec=last_init_times_unix_sec,
        nwp_model_names=nwp_model_names
    )

    error_checking.assert_equals(len(init_times_unix_sec), 1)
    init_time_unix_sec = init_times_unix_sec[0]

    if target_lag_times_hours is None:
        num_target_lag_times = 0
    else:
        num_target_lag_times = len(target_lag_times_hours)

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

    if patch_size_2pt5km_pixels is None:
        patch_location_dict = None
    else:
        patch_location_dict = misc_utils.determine_patch_locations(
            patch_size_2pt5km_pixels=patch_size_2pt5km_pixels,
            start_row_2pt5km=patch_start_row_2pt5km,
            start_column_2pt5km=patch_start_column_2pt5km
        )

    try:
        target_matrix = nn_utils.read_targets_one_example(
            init_time_unix_sec=init_time_unix_sec,
            target_lead_time_hours=target_lead_time_hours,
            target_field_names=target_field_names,
            target_dir_name=target_dir_name,
            target_norm_param_table_xarray=None,
            target_resid_norm_param_table_xarray=None,
            use_quantile_norm=False,
            patch_location_dict=patch_location_dict
        )
    except:
        warning_string = (
            'POTENTIAL ERROR: Could not read targets for init time {0:s}.  '
            'Something went wrong in `_read_targets_one_example`.'
        ).format(
            time_conversion.unix_sec_to_string(
                init_time_unix_sec, '%Y-%m-%d-%H'
            )
        )

        warnings.warn(warning_string)
        target_matrix = None

    if target_matrix is None:
        return None

    target_matrix = numpy.expand_dims(target_matrix, axis=0)

    if num_target_lag_times > 0:
        try:
            these_matrices = [
                nn_utils.read_targets_one_example(
                    init_time_unix_sec=init_time_unix_sec,
                    target_lead_time_hours=-1 * l,
                    target_field_names=target_field_names,
                    target_dir_name=target_dir_name,
                    target_norm_param_table_xarray=
                    target_norm_param_table_xarray,
                    target_resid_norm_param_table_xarray=
                    target_resid_norm_param_table_xarray,
                    use_quantile_norm=targets_use_quantile_norm,
                    patch_location_dict=patch_location_dict
                )
                for l in target_lag_times_hours
            ]

            predictor_matrix_lagged_targets = numpy.stack(
                these_matrices, axis=-2
            )
        except:
            warning_string = (
                'POTENTIAL ERROR: Could not read lagged targets for init '
                'time {0:s}.  Something went wrong in '
                '`_read_targets_one_example`.'
            ).format(
                time_conversion.unix_sec_to_string(
                    init_time_unix_sec, '%Y-%m-%d-%H'
                )
            )

            warnings.warn(warning_string)
            predictor_matrix_lagged_targets = None

        if predictor_matrix_lagged_targets is None:
            return None

        predictor_matrix_lagged_targets = numpy.expand_dims(
            predictor_matrix_lagged_targets, axis=0
        )
    else:
        predictor_matrix_lagged_targets = None

    need_baseline = do_residual_prediction or compare_to_baseline_in_loss

    if need_baseline:
        raw_baseline_matrix = None

        try:
            predictor_matrix_resid_baseline = (
                nwp_input.read_residual_baseline_one_example(
                    init_time_unix_sec=init_time_unix_sec,
                    nwp_model_name=resid_baseline_model_name,
                    nwp_lead_time_hours=resid_baseline_lead_time_hours,
                    nwp_directory_name=resid_baseline_model_dir_name,
                    target_field_names=target_field_names,
                    patch_location_dict=patch_location_dict,
                    predict_dewpoint_depression=True,
                    predict_gust_excess=True
                )
            )

            if compare_to_baseline_in_loss:
                tfn = target_field_names
                if (
                        urma_utils.DEWPOINT_2METRE_NAME in tfn
                        or urma_utils.WIND_GUST_10METRE_NAME in tfn
                ):
                    raw_baseline_matrix = (
                        nwp_input.read_residual_baseline_one_example(
                            init_time_unix_sec=init_time_unix_sec,
                            nwp_model_name=resid_baseline_model_name,
                            nwp_lead_time_hours=resid_baseline_lead_time_hours,
                            nwp_directory_name=resid_baseline_model_dir_name,
                            target_field_names=target_field_names,
                            patch_location_dict=patch_location_dict,
                            predict_dewpoint_depression=False,
                            predict_gust_excess=False
                        )
                    )
        except:
            warning_string = (
                'POTENTIAL ERROR: Could not read residual baseline for '
                'init time {0:s}.  Something went wrong in '
                '`nwp_input.read_residual_baseline_one_example`.'
            ).format(
                time_conversion.unix_sec_to_string(
                    init_time_unix_sec, '%Y-%m-%d-%H'
                )
            )

            warnings.warn(warning_string)
            predictor_matrix_resid_baseline = None

        if predictor_matrix_resid_baseline is None:
            return None

        predictor_matrix_resid_baseline = numpy.expand_dims(
            predictor_matrix_resid_baseline, axis=0
        )
        if raw_baseline_matrix is not None:
            raw_baseline_matrix = numpy.expand_dims(raw_baseline_matrix, axis=0)

        if compare_to_baseline_in_loss:
            if raw_baseline_matrix is None:
                target_matrix = numpy.concatenate(
                    [target_matrix, predictor_matrix_resid_baseline], axis=-1
                )
            else:
                target_matrix = numpy.concatenate(
                    [target_matrix, raw_baseline_matrix], axis=-1
                )
    else:
        predictor_matrix_resid_baseline = None

    try:
        (
            predictor_matrix_2pt5km,
            predictor_matrix_10km,
            predictor_matrix_20km,
            predictor_matrix_40km,
            found_any_predictors,
            found_all_predictors
        ) = nwp_input.read_predictors_one_example(
            init_time_unix_sec=init_time_unix_sec,
            nwp_model_names=nwp_model_names,
            nwp_lead_times_hours=nwp_lead_times_hours,
            nwp_model_to_field_names=nwp_model_to_field_names,
            nwp_model_to_dir_name=nwp_model_to_dir_name,
            nwp_norm_param_table_xarray=nwp_norm_param_table_xarray,
            nwp_resid_norm_param_table_xarray=nwp_resid_norm_param_table_xarray,
            use_quantile_norm=nwp_use_quantile_norm,
            backup_nwp_model_name=backup_nwp_model_name,
            backup_nwp_directory_name=backup_nwp_directory_name,
            patch_location_dict=patch_location_dict
        )
    except:
        warning_string = (
            'POTENTIAL ERROR: Could not read predictors for init time '
            '{0:s}.  Something went wrong in '
            '`nwp_input.read_predictors_one_example`.'
        ).format(
            time_conversion.unix_sec_to_string(
                init_time_unix_sec, '%Y-%m-%d-%H'
            )
        )

        warnings.warn(warning_string)
        predictor_matrix_2pt5km = None
        predictor_matrix_10km = None
        predictor_matrix_20km = None
        predictor_matrix_40km = None
        found_any_predictors = False
        found_all_predictors = False

    if not found_any_predictors:
        return None
    if require_all_predictors and not found_all_predictors:
        return None

    if predictor_matrix_2pt5km is not None:
        predictor_matrix_2pt5km = numpy.expand_dims(
            predictor_matrix_2pt5km, axis=0
        )
    if predictor_matrix_10km is not None:
        predictor_matrix_10km = numpy.expand_dims(predictor_matrix_10km, axis=0)
    if predictor_matrix_20km is not None:
        predictor_matrix_20km = numpy.expand_dims(predictor_matrix_20km, axis=0)
    if predictor_matrix_40km is not None:
        predictor_matrix_40km = numpy.expand_dims(predictor_matrix_40km, axis=0)

    if use_recent_biases:
        try:
            (
                recent_bias_matrix_2pt5km,
                recent_bias_matrix_10km,
                recent_bias_matrix_20km,
                recent_bias_matrix_40km,
                found_any_predictors,
                found_all_predictors
            ) = nwp_input.read_recent_biases_one_example(
                init_time_unix_sec=init_time_unix_sec,
                nwp_model_names=nwp_model_names,
                nwp_init_time_lags_hours=recent_bias_init_time_lags_hours,
                nwp_lead_times_hours=recent_bias_lead_times_hours,
                nwp_model_to_dir_name=nwp_model_to_dir_name,
                target_field_names=target_field_names,
                target_dir_name=target_dir_name,
                target_norm_param_table_xarray=target_norm_param_table_xarray,
                target_resid_norm_param_table_xarray=
                target_resid_norm_param_table_xarray,
                use_quantile_norm=targets_use_quantile_norm,
                backup_nwp_model_name=backup_nwp_model_name,
                backup_nwp_directory_name=backup_nwp_directory_name,
                patch_location_dict=patch_location_dict
            )
        except:
            warning_string = (
                'POTENTIAL ERROR: Could not read recent biases for init time '
                '{0:s}.  Something went wrong in '
                '`nwp_input.read_recent_biases_one_example`.'
            ).format(
                time_conversion.unix_sec_to_string(
                    init_time_unix_sec, '%Y-%m-%d-%H'
                )
            )

            warnings.warn(warning_string)
            recent_bias_matrix_2pt5km = None
            recent_bias_matrix_10km = None
            recent_bias_matrix_20km = None
            recent_bias_matrix_40km = None
            found_any_predictors = False
            found_all_predictors = False

        if not found_any_predictors:
            return None
        if require_all_predictors and not found_all_predictors:
            return None

        if recent_bias_matrix_2pt5km is not None:
            recent_bias_matrix_2pt5km = numpy.expand_dims(
                recent_bias_matrix_2pt5km, axis=0
            )
        if recent_bias_matrix_10km is not None:
            recent_bias_matrix_10km = numpy.expand_dims(
                recent_bias_matrix_10km, axis=0
            )
        if recent_bias_matrix_20km is not None:
            recent_bias_matrix_20km = numpy.expand_dims(
                recent_bias_matrix_20km, axis=0
            )
        if recent_bias_matrix_40km is not None:
            recent_bias_matrix_40km = numpy.expand_dims(
                recent_bias_matrix_40km, axis=0
            )
    else:
        recent_bias_matrix_2pt5km = None
        recent_bias_matrix_10km = None
        recent_bias_matrix_20km = None
        recent_bias_matrix_40km = None

    full_latitude_matrix_deg_n, full_longitude_matrix_deg_e = (
        nbm_utils.read_coords()
    )

    if patch_location_dict is None:
        nbm_constant_matrix = full_nbm_constant_matrix
        latitude_matrix_deg_n = full_latitude_matrix_deg_n
        longitude_matrix_deg_e = full_longitude_matrix_deg_e
    else:
        pld = patch_location_dict
        j_start = pld[misc_utils.ROW_LIMITS_2PT5KM_KEY][0]
        j_end = pld[misc_utils.ROW_LIMITS_2PT5KM_KEY][1] + 1
        k_start = pld[misc_utils.COLUMN_LIMITS_2PT5KM_KEY][0]
        k_end = pld[misc_utils.COLUMN_LIMITS_2PT5KM_KEY][1] + 1

        if full_nbm_constant_matrix is None:
            nbm_constant_matrix = None
        else:
            nbm_constant_matrix = (
                full_nbm_constant_matrix[j_start:j_end, k_start:k_end, ...]
            )

        latitude_matrix_deg_n = (
            full_latitude_matrix_deg_n[j_start:j_end, k_start:k_end]
        )
        longitude_matrix_deg_e = (
            full_longitude_matrix_deg_e[j_start:j_end, k_start:k_end]
        )

    latitude_matrix_deg_n = numpy.expand_dims(latitude_matrix_deg_n, axis=0)
    longitude_matrix_deg_e = numpy.expand_dims(longitude_matrix_deg_e, axis=0)
    if nbm_constant_matrix is not None:
        nbm_constant_matrix = numpy.expand_dims(nbm_constant_matrix, axis=0)

    target_matrix = numpy.concatenate(
        [target_matrix, mask_matrix_for_loss], axis=-1
    )

    predictor_matrices = nn_utils.create_data_dict_or_tuple(
        predictor_matrix_2pt5km=predictor_matrix_2pt5km,
        nbm_constant_matrix=nbm_constant_matrix,
        predictor_matrix_lagged_targets=predictor_matrix_lagged_targets,
        predictor_matrix_10km=predictor_matrix_10km,
        predictor_matrix_20km=predictor_matrix_20km,
        predictor_matrix_40km=predictor_matrix_40km,
        recent_bias_matrix_2pt5km=recent_bias_matrix_2pt5km,
        recent_bias_matrix_10km=recent_bias_matrix_10km,
        recent_bias_matrix_20km=recent_bias_matrix_20km,
        recent_bias_matrix_40km=recent_bias_matrix_40km,
        predictor_matrix_resid_baseline=predictor_matrix_resid_baseline,
        target_matrix=target_matrix,
        sentinel_value=sentinel_value,
        return_predictors_as_dict=return_predictors_as_dict,
        allow_nan=patch_location_dict is None
    )

    return {
        PREDICTOR_MATRICES_KEY: list(predictor_matrices),
        TARGET_MATRIX_KEY: target_matrix,
        INIT_TIMES_KEY: numpy.full(1, init_time_unix_sec),
        LATITUDE_MATRIX_KEY: latitude_matrix_deg_n,
        LONGITUDE_MATRIX_KEY: longitude_matrix_deg_e
    }


def data_generator_from_example_files(
        example_dir_name, first_init_times_unix_sec, last_init_times_unix_sec,
        num_examples_per_batch, return_predictors_as_dict=False):
    """Generates training or validation data from pre-processed .npz files.

    :param example_dir_name: Path to directory with pre-processed .npz files.
        Files will be found by `example_io.find_file` and read by
        `example_io.read_file`.
    :param first_init_times_unix_sec: length-P numpy array (where P = number of
        continuous periods in dataset), containing start time of each continuous
        period.
    :param last_init_times_unix_sec: length-P numpy array (where P = number of
        continuous periods in dataset), containing end time of each continuous
        period.
    :param num_examples_per_batch: Number of data examples per batch, usually
        just called "batch size".
    :param return_predictors_as_dict: See documentation for
        `neural_net_utils.create_data_dict_or_tuple`.
    :return: predictor_matrices: See documentation for `data_generator`.
    :return: target_matrix: Same.
    """

    # Check input args.
    error_checking.assert_is_numpy_array(
        first_init_times_unix_sec, num_dimensions=1
    )
    error_checking.assert_is_integer_numpy_array(first_init_times_unix_sec)
    num_periods = len(first_init_times_unix_sec)
    expected_dim = numpy.array([num_periods], dtype=int)

    error_checking.assert_is_numpy_array(
        last_init_times_unix_sec, exact_dimensions=expected_dim
    )
    error_checking.assert_is_integer_numpy_array(last_init_times_unix_sec)
    error_checking.assert_is_geq_numpy_array(
        last_init_times_unix_sec - first_init_times_unix_sec,
        0
    )

    error_checking.assert_is_integer(num_examples_per_batch)
    error_checking.assert_is_geq(num_examples_per_batch, 1)
    error_checking.assert_is_boolean(return_predictors_as_dict)

    # Do actual stuff.
    init_times_unix_sec = nn_utils.find_relevant_init_times(
        first_time_by_period_unix_sec=first_init_times_unix_sec,
        last_time_by_period_unix_sec=last_init_times_unix_sec,
        nwp_model_names=[nwp_model_utils.HRRR_MODEL_NAME]
    )
    numpy.random.shuffle(init_times_unix_sec)
    init_time_index = 0

    while True:
        num_examples_in_memory = 0
        predictor_matrices = None
        target_matrix = None

        while num_examples_in_memory < num_examples_per_batch:
            example_file_name = example_io.find_file(
                directory_name=example_dir_name,
                init_time_unix_sec=init_times_unix_sec[init_time_index],
                raise_error_if_missing=False
            )

            if not os.path.isfile(example_file_name):
                init_time_index, init_times_unix_sec = (
                    nn_utils.increment_init_time(
                        current_index=init_time_index,
                        init_times_unix_sec=init_times_unix_sec
                    )
                )
                continue

            print('Reading data from: "{0:s}"...'.format(example_file_name))
            these_predictor_matrices, this_target_matrix = example_io.read_file(
                example_file_name
            )

            if predictor_matrices is None:
                predictor_matrices = [
                    numpy.full(
                        (num_examples_per_batch,) + pm.shape[1:], numpy.nan
                    ) for pm in these_predictor_matrices
                ]
                target_matrix = numpy.full(
                    (num_examples_per_batch,) + this_target_matrix.shape[1:],
                    numpy.nan
                )

            i = num_examples_in_memory + 0
            target_matrix[i, ...] = this_target_matrix[0, ...]

            for j in range(len(predictor_matrices)):
                predictor_matrices[j][i, ...] = (
                    these_predictor_matrices[j][0, ...]
                )

        init_time_index, init_times_unix_sec = (
            nn_utils.increment_init_time(
                current_index=init_time_index,
                init_times_unix_sec=init_times_unix_sec
            )
        )
        yield predictor_matrices, target_matrix


def data_generator(option_dict, return_predictors_as_dict=False):
    """Generates training or validation data for neural network.

    E = number of examples per batch = "batch size"
    M = number of rows at 2.5-km resolution
    N = number of columns at 2.5-km resolution
    P = number of NWP fields (predictor variables) at 2.5-km resolution
    C = number of constant fields (at 2.5-km resolution)
    L = number of NWP lead times
    l = number of lag times for target fields used in predictors
    F = number of target fields
    B = number of lag times for recent NWP bias

    m = number of rows at 10-km resolution
    n = number of columns at 10-km resolution
    p = number of NWP fields at 10-km resolution
    mm = number of rows at 20-km resolution
    nn = number of columns at 20-km resolution
    pp = number of NWP fields at 20-km resolution
    mmm = number of rows at 40-km resolution
    nnn = number of columns at 40-km resolution
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
        `nwp_model_io.read_normalization_file`).
    option_dict["nwp_resid_norm_file_name"]: Path to file with
        residual-normalization params for NWP data (readable by
        `nwp_model_io.read_normalization_file`).  If you want just basic
        (z-score or quantile) normalization, make this None.
    option_dict["nwp_use_quantile_norm"]: Boolean flag.  If True, basic
        (non-residual) normalization will be a two-step procedure: first to
        quantiles, then to z-scores.  If False, basic normalization will go
        straight to z-scores, omitting the conversion to quantile scores.
    option_dict["backup_nwp_model_name"]: Name of backup model, used to fill
        missing data.
    option_dict["backup_nwp_directory_name"]: Directory for backup model.  Files
        therein will be found by `interp_nwp_model_io.find_file`.
    option_dict["target_lead_time_hours"]: Lead time for target fields.
    option_dict["target_field_names"]: length-F list with names of target
        fields.  Each must be accepted by `urma_utils.check_field_name`.
    option_dict["target_lag_times_hours"]: length-l numpy array of lag times for
        target fields used in predictors.  If you do not want to include lagged
        targets in the predictors, make this None.
    option_dict["target_dir_name"]: Path to directory with target fields (i.e.,
        URMA data).  Files within this directory will be found by
        `urma_io.find_file` and read by `urma_io.read_file`.
    option_dict["target_normalization_file_name"]: Path to file with
        normalization params for lagged target fields in predictors (readable by
        `urma_io.read_normalization_file`).
    option_dict["target_resid_norm_file_name"]: Path to file with
        residual-normalization params for lagged target fields in predictors
        (readable by `urma_io.read_normalization_file`).  If you want just basic
        (z-score or quantile) normalization, make this None.
    option_dict["targets_use_quantile_norm"]: Same as "nwp_use_quantile_norm"
        but for target fields.
    option_dict["recent_bias_init_time_lags_hours"]: length-B numpy array of lag
        times for recent NWP bias.  If you do not want predictors to include
        recent NWP bias, make this None.
    option_dict["recent_bias_lead_times_hours"]: length-B numpy array of lead
        times for recent NWP bias.  If you do not want predictors to include
        recent NWP bias, make this None.
    option_dict["nbm_constant_field_names"]: length-C list with names of NBM
        constant fields, to be used as predictors.  Each must be accepted by
        `nbm_constant_utils.check_field_name`.  If you do not want NBM-constant
        predictors, make this an empty list.
    option_dict["nbm_constant_file_name"]: Path to file with NBM constant
        fields (readable by `nbm_constant_io.read_file`).  If you do not want
        NBM-constant predictors, make this None.
    option_dict["compare_to_baseline_in_loss"]: Boolean flag.  If True, the loss
        function involves comparing to the residual baseline.  In other words,
        the loss function involves a skill score, except with the residual
        baseline instead of climo.
    option_dict["num_examples_per_batch"]: Number of data examples per batch,
        usually just called "batch size".
    option_dict["sentinel_value"]: All NaN will be replaced with this value.
    option_dict["patch_size_2pt5km_pixels"]: Patch size, in units of 2.5-km
        pixels.  For example, if patch_size_2pt5km_pixels = 448, then grid
        dimensions at the finest resolution (2.5 km) are 448 x 448.  If you want
        to train with full-grid data, make this None.
    option_dict["patch_buffer_size_2pt5km_pixels"]:
        (used only if "patch_size_2pt5km_pixels" is not None)
        Buffer between the outer domain (used for predictors) and the inner
        domain (used to penalize predictions in loss function).  This must be a
        non-negative integer.
    option_dict["patch_start_row_2pt5km"]:
        (used only if "patch_size_2pt5km_pixels" is not None)
        Start row for patches.  If you want patches to come from random
        locations instead, make this None.
    option_dict["patch_start_column_2pt5km"]: Same but for start column.
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

    :param return_predictors_as_dict: See documentation for
        `neural_net_utils.create_data_dict_or_tuple`.

    :return: predictor_matrices: List with the following items.  Some items may
        be missing.

    predictor_matrices[0]: E-by-M-by-N-by-L-by-P numpy array of predictors at
        2.5-km resolution.
    predictor_matrices[1]: E-by-M-by-N-by-C numpy array of NBM-constant
        predictors, also at 2.5-km resolution.
    predictor_matrices[2]: E-by-M-by-N-by-l-by-F numpy array of lagged targets
        at 2.5-km resolution.
    predictor_matrices[3]: E-by-m-by-n-by-L-by-p numpy array of predictors at
        10-km resolution.
    predictor_matrices[4]: E-by-mm-by-nn-by-L-by-pp numpy array of predictors at
        20-km resolution.
    predictor_matrices[5]: E-by-mmm-by-nnn-by-L-by-ppp numpy array of predictors
        at 40-km resolution.
    predictor_matrices[6]: E-by-M-by-N-by-B-by-? numpy array of recent NWP
        biases at 2.5-km resolution.
    predictor_matrices[7]: E-by-m-by-n-by-B-by-? numpy array of recent NWP
        biases at 10-km resolution.
    predictor_matrices[8]: E-by-mm-by-nn-by-B-by-? numpy array of recent NWP
        biases at 20-km resolution.
    predictor_matrices[9]: E-by-mmm-by-nnn-by-B-by-? numpy array of recent NWP
        biases at 40-km resolution.
    predictor_matrices[10]: E-by-M-by-N-by-F numpy array of baseline values for
        residual prediction.

    :return: target_matrix: If `compare_to_baseline_in_loss == False`, this is
        an E-by-M-by-N-by-(F + 1) numpy array of targets at 2.5-km resolution.
        The first F channels are actual target values; the last channel is a
        binary mask, where 1 (0) indicates that the pixel should (not) be
        considered in the loss function.

    If `compare_to_baseline_in_loss == True`, this is an E-by-M-by-N-by-(2F + 1)
        numpy array, where target_matrix[:F] contains actual values of the
        target fields; target_matrix[F:-1] contains baseline-forecast values of
        the target fields; and target_matrix[..., -1] contains the binary mask.
    """

    option_dict = nn_utils.check_generator_args(option_dict)
    first_init_times_unix_sec = option_dict[nn_utils.FIRST_INIT_TIMES_KEY]
    last_init_times_unix_sec = option_dict[nn_utils.LAST_INIT_TIMES_KEY]
    nwp_lead_times_hours = option_dict[nn_utils.NWP_LEAD_TIMES_KEY]
    nwp_model_to_dir_name = option_dict[nn_utils.NWP_MODEL_TO_DIR_KEY]
    nwp_model_to_field_names = option_dict[nn_utils.NWP_MODEL_TO_FIELDS_KEY]
    nwp_normalization_file_name = option_dict[nn_utils.NWP_NORM_FILE_KEY]
    nwp_resid_norm_file_name = option_dict[nn_utils.NWP_RESID_NORM_FILE_KEY]
    nwp_use_quantile_norm = option_dict[nn_utils.NWP_USE_QUANTILE_NORM_KEY]
    backup_nwp_model_name = option_dict[nn_utils.BACKUP_NWP_MODEL_KEY]
    backup_nwp_directory_name = option_dict[nn_utils.BACKUP_NWP_DIR_KEY]
    target_lead_time_hours = option_dict[nn_utils.TARGET_LEAD_TIME_KEY]
    target_field_names = option_dict[nn_utils.TARGET_FIELDS_KEY]
    target_lag_times_hours = option_dict[nn_utils.TARGET_LAG_TIMES_KEY]
    target_dir_name = option_dict[nn_utils.TARGET_DIR_KEY]
    target_normalization_file_name = option_dict[nn_utils.TARGET_NORM_FILE_KEY]
    target_resid_norm_file_name = option_dict[
        nn_utils.TARGET_RESID_NORM_FILE_KEY
    ]
    targets_use_quantile_norm = option_dict[
        nn_utils.TARGETS_USE_QUANTILE_NORM_KEY
    ]
    recent_bias_init_time_lags_hours = option_dict[
        nn_utils.RECENT_BIAS_LAG_TIMES_KEY
    ]
    recent_bias_lead_times_hours = option_dict[
        nn_utils.RECENT_BIAS_LEAD_TIMES_KEY
    ]
    nbm_constant_field_names = option_dict[nn_utils.NBM_CONSTANT_FIELDS_KEY]
    nbm_constant_file_name = option_dict[nn_utils.NBM_CONSTANT_FILE_KEY]
    compare_to_baseline_in_loss = option_dict[
        nn_utils.COMPARE_TO_BASELINE_IN_LOSS_KEY
    ]
    num_examples_per_batch = option_dict[nn_utils.BATCH_SIZE_KEY]
    sentinel_value = option_dict[nn_utils.SENTINEL_VALUE_KEY]
    patch_size_2pt5km_pixels = option_dict[nn_utils.PATCH_SIZE_KEY]
    patch_buffer_size_2pt5km_pixels = option_dict[
        nn_utils.PATCH_BUFFER_SIZE_KEY
    ]
    patch_start_row_2pt5km = option_dict[nn_utils.PATCH_START_ROW_KEY]
    patch_start_column_2pt5km = option_dict[nn_utils.PATCH_START_COLUMN_KEY]
    require_all_predictors = option_dict[nn_utils.REQUIRE_ALL_PREDICTORS_KEY]

    do_residual_prediction = option_dict[nn_utils.DO_RESIDUAL_PREDICTION_KEY]
    resid_baseline_model_name = option_dict[nn_utils.RESID_BASELINE_MODEL_KEY]
    resid_baseline_model_dir_name = option_dict[
        nn_utils.RESID_BASELINE_MODEL_DIR_KEY
    ]
    resid_baseline_lead_time_hours = option_dict[
        nn_utils.RESID_BASELINE_LEAD_TIME_KEY
    ]

    use_recent_biases = not (
            recent_bias_init_time_lags_hours is None
            or recent_bias_lead_times_hours is None
    )
    if use_recent_biases:
        num_recent_bias_times = len(recent_bias_init_time_lags_hours)
    else:
        num_recent_bias_times = 0

    if patch_size_2pt5km_pixels is None:
        num_rows, num_columns = nwp_input.get_grid_dimensions(
            grid_spacing_km=2.5, patch_location_dict=None
        )
        mask_matrix_for_loss = numpy.full((num_rows, num_columns), 1, dtype=int)
    else:
        mask_matrix_for_loss = nn_utils.patch_buffer_to_mask(
            patch_size_2pt5km_pixels=patch_size_2pt5km_pixels,
            patch_buffer_size_2pt5km_pixels=patch_buffer_size_2pt5km_pixels
        )

    mask_matrix_for_loss = numpy.repeat(
        numpy.expand_dims(mask_matrix_for_loss, axis=0),
        repeats=num_examples_per_batch,
        axis=0
    )
    mask_matrix_for_loss = numpy.expand_dims(mask_matrix_for_loss, axis=-1)

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

    if nwp_resid_norm_file_name is None:
        nwp_resid_norm_param_table_xarray = None
    else:
        print('Reading residual-normalization params from: "{0:s}"...'.format(
            nwp_resid_norm_file_name
        ))
        nwp_resid_norm_param_table_xarray = (
            nwp_model_io.read_normalization_file(nwp_resid_norm_file_name)
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

    if target_resid_norm_file_name is None:
        target_resid_norm_param_table_xarray = None
    else:
        print('Reading residual-normalization params from: "{0:s}"...'.format(
            target_resid_norm_file_name
        ))
        target_resid_norm_param_table_xarray = urma_io.read_normalization_file(
            target_resid_norm_file_name
        )

    init_times_unix_sec = nn_utils.find_relevant_init_times(
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

    num_target_fields = len(target_field_names)
    if target_lag_times_hours is None:
        num_target_lag_times = 0
    else:
        num_target_lag_times = len(target_lag_times_hours)

    while True:
        if patch_size_2pt5km_pixels is None:
            patch_location_dict = None
        else:
            patch_location_dict = misc_utils.determine_patch_locations(
                patch_size_2pt5km_pixels=patch_size_2pt5km_pixels,
                start_row_2pt5km=patch_start_row_2pt5km,
                start_column_2pt5km=patch_start_column_2pt5km
            )

        matrix_dict = nn_utils.init_matrices_1batch(
            nwp_model_names=nwp_model_names,
            nwp_model_to_field_names=nwp_model_to_field_names,
            num_nwp_lead_times=len(nwp_lead_times_hours),
            target_field_names=target_field_names,
            num_recent_bias_times=num_recent_bias_times,
            num_target_lag_times=num_target_lag_times,
            num_examples_per_batch=num_examples_per_batch,
            patch_location_dict=patch_location_dict,
            do_residual_prediction=do_residual_prediction
        )
        predictor_matrix_2pt5km = matrix_dict[
            nn_utils.PREDICTOR_MATRIX_2PT5KM_KEY
        ]
        predictor_matrix_10km = matrix_dict[nn_utils.PREDICTOR_MATRIX_10KM_KEY]
        predictor_matrix_20km = matrix_dict[nn_utils.PREDICTOR_MATRIX_20KM_KEY]
        predictor_matrix_40km = matrix_dict[nn_utils.PREDICTOR_MATRIX_40KM_KEY]
        recent_bias_matrix_2pt5km = matrix_dict[
            nn_utils.RECENT_BIAS_MATRIX_2PT5KM_KEY
        ]
        recent_bias_matrix_10km = matrix_dict[
            nn_utils.RECENT_BIAS_MATRIX_10KM_KEY
        ]
        recent_bias_matrix_20km = matrix_dict[
            nn_utils.RECENT_BIAS_MATRIX_20KM_KEY
        ]
        recent_bias_matrix_40km = matrix_dict[
            nn_utils.RECENT_BIAS_MATRIX_40KM_KEY
        ]
        predictor_matrix_resid_baseline = matrix_dict[
            nn_utils.PREDICTOR_MATRIX_BASELINE_KEY
        ]
        predictor_matrix_lagged_targets = matrix_dict[
            nn_utils.PREDICTOR_MATRIX_LAGTGT_KEY
        ]
        target_matrix = matrix_dict[nn_utils.TARGET_MATRIX_KEY]

        if compare_to_baseline_in_loss:
            new_dims = target_matrix.shape[:-1] + (2 * num_target_fields,)
            target_matrix = numpy.full(new_dims, numpy.nan)

        if full_nbm_constant_matrix is None:
            nbm_constant_matrix = None
        else:
            these_dims = (
                target_matrix.shape[:-1] + (len(nbm_constant_field_names),)
            )
            nbm_constant_matrix = numpy.full(these_dims, numpy.nan)

        num_examples_in_memory = 0

        while num_examples_in_memory < num_examples_per_batch:
            if patch_size_2pt5km_pixels is None:
                patch_location_dict = None
            else:
                patch_location_dict = misc_utils.determine_patch_locations(
                    patch_size_2pt5km_pixels=patch_size_2pt5km_pixels,
                    start_row_2pt5km=patch_start_row_2pt5km,
                    start_column_2pt5km=patch_start_column_2pt5km
                )

            i = num_examples_in_memory + 0

            try:
                this_target_matrix = nn_utils.read_targets_one_example(
                    init_time_unix_sec=init_times_unix_sec[init_time_index],
                    target_lead_time_hours=target_lead_time_hours,
                    target_field_names=target_field_names,
                    target_dir_name=target_dir_name,
                    target_norm_param_table_xarray=None,
                    target_resid_norm_param_table_xarray=None,
                    use_quantile_norm=False,
                    patch_location_dict=patch_location_dict
                )

                if numpy.any(numpy.isnan(this_target_matrix)):
                    this_target_matrix = None
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
                init_time_index, init_times_unix_sec = (
                    nn_utils.increment_init_time(
                        current_index=init_time_index,
                        init_times_unix_sec=init_times_unix_sec
                    )
                )
                continue

            target_matrix[i, ..., :num_target_fields] = this_target_matrix

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

            if num_target_lag_times > 0:
                try:
                    these_matrices = [
                        nn_utils.read_targets_one_example(
                            init_time_unix_sec=
                            init_times_unix_sec[init_time_index],
                            target_lead_time_hours=-1 * l,
                            target_field_names=target_field_names,
                            target_dir_name=target_dir_name,
                            target_norm_param_table_xarray=
                            target_norm_param_table_xarray,
                            target_resid_norm_param_table_xarray=
                            target_resid_norm_param_table_xarray,
                            use_quantile_norm=targets_use_quantile_norm,
                            patch_location_dict=patch_location_dict
                        )
                        for l in target_lag_times_hours
                    ]

                    this_predictor_matrix_lagged_targets = numpy.stack(
                        these_matrices, axis=-2
                    )
                except:
                    warning_string = (
                        'POTENTIAL ERROR: Could not read lagged targets for '
                        'init time {0:s}.  Something went wrong in '
                        '`_read_targets_one_example`.'
                    ).format(
                        time_conversion.unix_sec_to_string(
                            init_times_unix_sec[init_time_index], '%Y-%m-%d-%H'
                        )
                    )

                    warnings.warn(warning_string)
                    this_predictor_matrix_lagged_targets = None

                if this_predictor_matrix_lagged_targets is None:
                    init_time_index, init_times_unix_sec = (
                        nn_utils.increment_init_time(
                            current_index=init_time_index,
                            init_times_unix_sec=init_times_unix_sec
                        )
                    )
                    continue

                predictor_matrix_lagged_targets[i, ...] = (
                    this_predictor_matrix_lagged_targets
                )

            need_baseline = (
                    do_residual_prediction or compare_to_baseline_in_loss
            )

            if need_baseline:
                this_raw_baseline_matrix = None

                try:
                    this_baseline_matrix = (
                        nwp_input.read_residual_baseline_one_example(
                            init_time_unix_sec=
                            init_times_unix_sec[init_time_index],
                            nwp_model_name=resid_baseline_model_name,
                            nwp_lead_time_hours=resid_baseline_lead_time_hours,
                            nwp_directory_name=resid_baseline_model_dir_name,
                            target_field_names=target_field_names,
                            patch_location_dict=patch_location_dict,
                            predict_dewpoint_depression=True,
                            predict_gust_excess=True
                        )
                    )

                    if numpy.any(numpy.isnan(this_baseline_matrix)):
                        this_baseline_matrix = None

                    if compare_to_baseline_in_loss:
                        tfn = target_field_names
                        if (
                                urma_utils.DEWPOINT_2METRE_NAME in tfn
                                or urma_utils.WIND_GUST_10METRE_NAME in tfn
                        ):
                            this_raw_baseline_matrix = (
                                nwp_input.read_residual_baseline_one_example(
                                    init_time_unix_sec=
                                    init_times_unix_sec[init_time_index],
                                    nwp_model_name=resid_baseline_model_name,
                                    nwp_lead_time_hours=
                                    resid_baseline_lead_time_hours,
                                    nwp_directory_name=
                                    resid_baseline_model_dir_name,
                                    target_field_names=target_field_names,
                                    patch_location_dict=patch_location_dict,
                                    predict_dewpoint_depression=False,
                                    predict_gust_excess=False
                                )
                            )
                except:
                    warning_string = (
                        'POTENTIAL ERROR: Could not read residual baseline for '
                        'init time {0:s}.  Something went wrong in '
                        '`nwp_input.read_residual_baseline_one_example`.'
                    ).format(
                        time_conversion.unix_sec_to_string(
                            init_times_unix_sec[init_time_index], '%Y-%m-%d-%H'
                        )
                    )

                    warnings.warn(warning_string)
                    this_baseline_matrix = None

                if this_baseline_matrix is None:
                    init_time_index, init_times_unix_sec = (
                        nn_utils.increment_init_time(
                            current_index=init_time_index,
                            init_times_unix_sec=init_times_unix_sec
                        )
                    )
                    continue

                predictor_matrix_resid_baseline[i, ...] = this_baseline_matrix

                if compare_to_baseline_in_loss:
                    if this_raw_baseline_matrix is None:
                        target_matrix[i, ..., num_target_fields:] = (
                            this_baseline_matrix
                        )
                    else:
                        target_matrix[i, ..., num_target_fields:] = (
                            this_raw_baseline_matrix
                        )

            try:
                (
                    this_predictor_matrix_2pt5km,
                    this_predictor_matrix_10km,
                    this_predictor_matrix_20km,
                    this_predictor_matrix_40km,
                    found_any_predictors,
                    found_all_predictors
                ) = nwp_input.read_predictors_one_example(
                    init_time_unix_sec=init_times_unix_sec[init_time_index],
                    nwp_model_names=nwp_model_names,
                    nwp_lead_times_hours=nwp_lead_times_hours,
                    nwp_model_to_field_names=nwp_model_to_field_names,
                    nwp_model_to_dir_name=nwp_model_to_dir_name,
                    nwp_norm_param_table_xarray=nwp_norm_param_table_xarray,
                    nwp_resid_norm_param_table_xarray=
                    nwp_resid_norm_param_table_xarray,
                    use_quantile_norm=nwp_use_quantile_norm,
                    patch_location_dict=patch_location_dict,
                    backup_nwp_model_name=backup_nwp_model_name,
                    backup_nwp_directory_name=backup_nwp_directory_name
                )
            except:
                warning_string = (
                    'POTENTIAL ERROR: Could not read predictors for init '
                    'time {0:s}.  Something went wrong in '
                    '`nwp_input.read_predictors_one_example`.'
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
                found_all_predictors = False

            if not found_any_predictors:
                init_time_index, init_times_unix_sec = (
                    nn_utils.increment_init_time(
                        current_index=init_time_index,
                        init_times_unix_sec=init_times_unix_sec
                    )
                )
                continue

            if require_all_predictors and not found_all_predictors:
                init_time_index, init_times_unix_sec = (
                    nn_utils.increment_init_time(
                        current_index=init_time_index,
                        init_times_unix_sec=init_times_unix_sec
                    )
                )
                continue

            if predictor_matrix_2pt5km is not None:
                predictor_matrix_2pt5km[i, ...] = (
                    this_predictor_matrix_2pt5km
                )
            if predictor_matrix_10km is not None:
                predictor_matrix_10km[i, ...] = this_predictor_matrix_10km
            if predictor_matrix_20km is not None:
                predictor_matrix_20km[i, ...] = this_predictor_matrix_20km
            if predictor_matrix_40km is not None:
                predictor_matrix_40km[i, ...] = this_predictor_matrix_40km

            if use_recent_biases:
                try:
                    (
                        this_recent_bias_matrix_2pt5km,
                        this_recent_bias_matrix_10km,
                        this_recent_bias_matrix_20km,
                        this_recent_bias_matrix_40km,
                        found_any_predictors,
                        found_all_predictors
                    ) = nwp_input.read_recent_biases_one_example(
                        init_time_unix_sec=init_times_unix_sec[init_time_index],
                        nwp_model_names=nwp_model_names,
                        nwp_init_time_lags_hours=
                        recent_bias_init_time_lags_hours,
                        nwp_lead_times_hours=recent_bias_lead_times_hours,
                        nwp_model_to_dir_name=nwp_model_to_dir_name,
                        target_field_names=target_field_names,
                        target_dir_name=target_dir_name,
                        target_norm_param_table_xarray=
                        target_norm_param_table_xarray,
                        target_resid_norm_param_table_xarray=
                        target_resid_norm_param_table_xarray,
                        use_quantile_norm=targets_use_quantile_norm,
                        backup_nwp_model_name=backup_nwp_model_name,
                        backup_nwp_directory_name=backup_nwp_directory_name,
                        patch_location_dict=patch_location_dict
                    )
                except:
                    warning_string = (
                        'POTENTIAL ERROR: Could not read recent biases for '
                        'init time {0:s}.  Something went wrong in '
                        '`nwp_input.read_recent_biases_one_example`.'
                    ).format(
                        time_conversion.unix_sec_to_string(
                            init_times_unix_sec[init_time_index], '%Y-%m-%d-%H'
                        )
                    )

                    warnings.warn(warning_string)
                    this_recent_bias_matrix_2pt5km = None
                    this_recent_bias_matrix_10km = None
                    this_recent_bias_matrix_20km = None
                    this_recent_bias_matrix_40km = None
                    found_any_predictors = False
                    found_all_predictors = False

                if not found_any_predictors:
                    init_time_index, init_times_unix_sec = (
                        nn_utils.increment_init_time(
                            current_index=init_time_index,
                            init_times_unix_sec=init_times_unix_sec
                        )
                    )
                    continue

                if require_all_predictors and not found_all_predictors:
                    init_time_index, init_times_unix_sec = (
                        nn_utils.increment_init_time(
                            current_index=init_time_index,
                            init_times_unix_sec=init_times_unix_sec
                        )
                    )
                    continue

                if recent_bias_matrix_2pt5km is not None:
                    recent_bias_matrix_2pt5km[i, ...] = (
                        this_recent_bias_matrix_2pt5km
                    )
                if recent_bias_matrix_10km is not None:
                    recent_bias_matrix_10km[i, ...] = (
                        this_recent_bias_matrix_10km
                    )
                if recent_bias_matrix_20km is not None:
                    recent_bias_matrix_20km[i, ...] = (
                        this_recent_bias_matrix_20km
                    )
                if recent_bias_matrix_40km is not None:
                    recent_bias_matrix_40km[i, ...] = (
                        this_recent_bias_matrix_40km
                    )

            num_examples_in_memory += 1
            init_time_index, init_times_unix_sec = (
                nn_utils.increment_init_time(
                    current_index=init_time_index,
                    init_times_unix_sec=init_times_unix_sec
                )
            )

        target_matrix = numpy.concatenate(
            [target_matrix, mask_matrix_for_loss], axis=-1
        )

        predictor_matrices = nn_utils.create_data_dict_or_tuple(
            predictor_matrix_2pt5km=predictor_matrix_2pt5km,
            nbm_constant_matrix=nbm_constant_matrix,
            predictor_matrix_lagged_targets=predictor_matrix_lagged_targets,
            predictor_matrix_10km=predictor_matrix_10km,
            predictor_matrix_20km=predictor_matrix_20km,
            predictor_matrix_40km=predictor_matrix_40km,
            recent_bias_matrix_2pt5km=recent_bias_matrix_2pt5km,
            recent_bias_matrix_10km=recent_bias_matrix_10km,
            recent_bias_matrix_20km=recent_bias_matrix_20km,
            recent_bias_matrix_40km=recent_bias_matrix_40km,
            predictor_matrix_resid_baseline=predictor_matrix_resid_baseline,
            target_matrix=target_matrix,
            sentinel_value=sentinel_value,
            return_predictors_as_dict=return_predictors_as_dict
        )

        yield predictor_matrices, target_matrix


def train_model(
        model_object, num_epochs, use_exp_moving_average_with_decay,
        num_training_batches_per_epoch, training_option_dict,
        num_validation_batches_per_epoch, validation_option_dict,
        loss_function_string, optimizer_function_string,
        metric_function_strings,
        u_net_architecture_dict, chiu_net_architecture_dict,
        chiu_net_pp_architecture_dict, chiu_next_pp_architecture_dict,
        plateau_patience_epochs, plateau_learning_rate_multiplier,
        early_stopping_patience_epochs, output_dir_name,
        training_generator=None, validation_generator=None):
    """Trains neural net with generator.

    :param model_object: Untrained neural net (instance of
        `keras.models.Model`).
    :param num_epochs: Number of training epochs.
    :param use_exp_moving_average_with_decay: Decay parameter for EMA
        (exponential moving average) training method.  If you do not want to use
        EMA, make this None.
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
    :param u_net_architecture_dict: Dictionary with architecture options for
        `u_net_architecture.create_model`.  If the model being trained is not
        a U-net, make this None.
    :param chiu_net_architecture_dict: Dictionary with architecture options for
        `chiu_net_architecture.create_model`.  If the model being trained is not
        a Chiu-net, make this None.
    :param chiu_net_pp_architecture_dict: Dictionary with architecture options
        for `chiu_net_pp_architecture.create_model`.  If the model being trained
        is not a Chiu-net++, make this None.
    :param chiu_next_pp_architecture_dict: Dictionary with architecture options
        for `chiu_next_pp_architecture.create_model`.  If the model being
        trained is not a Chiu-next++, make this None.
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
    :param training_generator: Leave this alone if you don't know what you're
        doing.
    :param validation_generator: Leave this alone if you don't know what you're
        doing.
    """

    file_system_utils.mkdir_recursive_if_necessary(
        directory_name=output_dir_name
    )

    backup_dir_name = '{0:s}/backup_and_restore'.format(output_dir_name)
    file_system_utils.mkdir_recursive_if_necessary(
        directory_name=backup_dir_name
    )

    if use_exp_moving_average_with_decay is not None:
        error_checking.assert_is_geq(use_exp_moving_average_with_decay, 0.5)
        error_checking.assert_is_less_than(
            use_exp_moving_average_with_decay, 1.
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
        nn_utils.FIRST_INIT_TIMES_KEY, nn_utils.LAST_INIT_TIMES_KEY,
        nn_utils.NWP_MODEL_TO_DIR_KEY, nn_utils.TARGET_DIR_KEY
    ]
    for this_key in list(training_option_dict.keys()):
        if this_key in validation_keys_to_keep:
            continue

        validation_option_dict[this_key] = training_option_dict[this_key]

    training_option_dict = nn_utils.check_generator_args(training_option_dict)
    validation_option_dict = nn_utils.check_generator_args(
        validation_option_dict
    )

    model_file_name = '{0:s}/model.weights.h5'.format(output_dir_name)
    history_file_name = '{0:s}/history.csv'.format(output_dir_name)

    try:
        history_table_pandas = pandas.read_csv(history_file_name)
        initial_epoch = history_table_pandas['epoch'].max() + 1
        best_validation_loss = history_table_pandas['val_loss'].min()
    except:
        initial_epoch = 0
        best_validation_loss = numpy.inf

    history_object = keras.callbacks.CSVLogger(
        filename=history_file_name, separator=',', append=True
    )
    checkpoint_object = keras.callbacks.ModelCheckpoint(
        filepath=model_file_name, monitor='val_loss', verbose=1,
        save_best_only=True, save_weights_only=True, mode='min',
        save_freq='epoch'
    )
    checkpoint_object.best = best_validation_loss

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
        backup_dir_name, save_freq='epoch', delete_checkpoint=False
    )

    list_of_callback_objects = [
        history_object, checkpoint_object,
        early_stopping_object, plateau_object,
        backup_object
    ]

    if training_generator is None or validation_generator is None:
        training_generator = data_generator(training_option_dict)
        validation_generator = data_generator(validation_option_dict)

    metafile_name = nn_utils.find_metafile(
        model_file_name=model_file_name, raise_error_if_missing=False
    )
    print('Writing metadata to: "{0:s}"...'.format(metafile_name))

    nn_utils.write_metafile(
        pickle_file_name=metafile_name,
        num_epochs=num_epochs,
        use_exp_moving_average_with_decay=use_exp_moving_average_with_decay,
        num_training_batches_per_epoch=num_training_batches_per_epoch,
        training_option_dict=training_option_dict,
        num_validation_batches_per_epoch=num_validation_batches_per_epoch,
        validation_option_dict=validation_option_dict,
        loss_function_string=loss_function_string,
        optimizer_function_string=optimizer_function_string,
        metric_function_strings=metric_function_strings,
        u_net_architecture_dict=u_net_architecture_dict,
        chiu_net_architecture_dict=chiu_net_architecture_dict,
        chiu_net_pp_architecture_dict=chiu_net_pp_architecture_dict,
        chiu_next_pp_architecture_dict=chiu_next_pp_architecture_dict,
        plateau_patience_epochs=plateau_patience_epochs,
        plateau_learning_rate_multiplier=plateau_learning_rate_multiplier,
        early_stopping_patience_epochs=early_stopping_patience_epochs,
        patch_overlap_fast_gen_2pt5km_pixels=None
    )

    if use_exp_moving_average_with_decay is None:
        model_object.fit(
            x=training_generator,
            steps_per_epoch=num_training_batches_per_epoch,
            epochs=num_epochs,
            initial_epoch=initial_epoch,
            verbose=1,
            callbacks=list_of_callback_objects,
            validation_data=validation_generator,
            validation_steps=num_validation_batches_per_epoch
        )
        return

    ema_object = nn_utils.EMAHelper(
        model=model_object,
        decay=use_exp_moving_average_with_decay
    )

    ema_backup_dir_name = '{0:s}/exponential_moving_average'.format(
        output_dir_name
    )
    file_system_utils.mkdir_recursive_if_necessary(
        directory_name=ema_backup_dir_name
    )

    ema_object.restore_optimizer_state(
        checkpoint_dir=ema_backup_dir_name,
        raise_error_if_missing=initial_epoch > 0
    )

    for this_epoch in range(initial_epoch, num_epochs):
        model_object.fit(
            x=training_generator,
            steps_per_epoch=num_training_batches_per_epoch,
            epochs=this_epoch + 1,
            initial_epoch=this_epoch,
            verbose=1,
            callbacks=list_of_callback_objects,
            validation_data=validation_generator,
            validation_steps=num_validation_batches_per_epoch
        )

        ema_object.apply_ema()
        ema_object.save_optimizer_state(
            checkpoint_dir=ema_backup_dir_name, epoch=this_epoch
        )


def apply_model(
        model_object, predictor_matrices, num_examples_per_batch,
        target_field_names, verbose=True):
    """Applies trained neural net -- inference time!

    E = number of examples
    M = number of rows in grid
    N = number of columns in grid
    F = number of target fields
    S = ensemble size

    :param model_object: Trained neural net (instance of `keras.models.Model`).
    :param predictor_matrices: See output doc for `data_generator`.
    :param num_examples_per_batch: Batch size.
    :param target_field_names: length-F list of target fields (each must be
        accepted by `urma_utils.check_field_name`).
    :param verbose: Boolean flag.  If True, will print progress messages.
    :return: prediction_matrix: E-by-M-by-N-by-F-by-S numpy array of predicted
        values.
    """

    # Check input args.
    for this_matrix in predictor_matrices:
        error_checking.assert_is_numpy_array_without_nan(this_matrix)

    error_checking.assert_is_integer(num_examples_per_batch)
    error_checking.assert_is_geq(num_examples_per_batch, 1)
    num_examples = predictor_matrices[0].shape[0]
    num_examples_per_batch = min([num_examples_per_batch, num_examples])

    error_checking.assert_is_string_list(target_field_names)
    for this_name in target_field_names:
        urma_utils.check_field_name(this_name)

    error_checking.assert_is_boolean(verbose)

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

    while len(prediction_matrix.shape) < 5:
        prediction_matrix = numpy.expand_dims(prediction_matrix, axis=-1)

    if urma_utils.DEWPOINT_2METRE_NAME in target_field_names:
        prediction_matrix = nn_utils.predicted_depression_to_dewpoint(
            prediction_matrix=prediction_matrix,
            target_field_names=target_field_names
        )

    if urma_utils.WIND_GUST_10METRE_NAME in target_field_names:
        prediction_matrix = nn_utils.predicted_gust_excess_to_speed(
            prediction_matrix=prediction_matrix,
            target_field_names=target_field_names
        )

    return prediction_matrix

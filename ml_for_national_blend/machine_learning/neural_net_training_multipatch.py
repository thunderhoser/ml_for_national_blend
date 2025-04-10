"""NN training and inference with multiple patches around the NBM domain."""

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
from ml_for_national_blend.outside_code import number_rounding
from ml_for_national_blend.outside_code import file_system_utils
from ml_for_national_blend.outside_code import error_checking
from ml_for_national_blend.machine_learning import neural_net_utils as nn_utils
from ml_for_national_blend.machine_learning import nwp_input
from ml_for_national_blend.machine_learning import \
    neural_net_training_simple as nn_training_simple

TOLERANCE = 1e-6
HOURS_TO_SECONDS = 3600
POSSIBLE_DOWNSAMPLING_FACTORS = numpy.array([1, 4, 8, 16], dtype=int)

PREDICTOR_MATRICES_KEY = 'predictor_matrices_key'
TARGET_MATRIX_KEY = 'target_matrix'
INIT_TIMES_KEY = 'init_times_unix_sec'
LATITUDE_MATRIX_KEY = 'latitude_matrix_deg_n'
LONGITUDE_MATRIX_KEY = 'longitude_matrix_deg_e'

NUM_FULL_ROWS_KEY = 'num_rows_in_full_grid'
NUM_FULL_COLUMNS_KEY = 'num_columns_in_full_grid'
NUM_PATCH_ROWS_KEY = 'num_rows_in_patch'
NUM_PATCH_COLUMNS_KEY = 'num_columns_in_patch'
PATCH_OVERLAP_SIZE_KEY = 'patch_overlap_size_2pt5km_pixels'
PATCH_START_ROW_KEY = 'patch_start_row_2pt5km'
PATCH_START_COLUMN_KEY = 'patch_start_column_2pt5km'


def make_trapezoidal_weight_matrix(patch_size_2pt5km_pixels,
                                   patch_overlap_size_2pt5km_pixels):
    """Creates trapezoidal weight matrix for applying NN to full grid.

    :param patch_size_2pt5km_pixels: See doc for
        `update_patch_metalocation_dict`.
    :param patch_overlap_size_2pt5km_pixels: Same.
    :return: trapezoidal_weight_matrix: M-by-M numpy array of weights, where
        M = patch size.
    """

    middle_length = patch_size_2pt5km_pixels - patch_overlap_size_2pt5km_pixels
    middle_start_index = (patch_size_2pt5km_pixels - middle_length) // 2

    weights_before_plateau = numpy.linspace(
        0, 1,
        num=middle_start_index, endpoint=False, dtype=float
    )
    weights_before_plateau = numpy.linspace(
        weights_before_plateau[1], 1,
        num=middle_start_index, endpoint=False, dtype=float
    )
    trapezoidal_weights = numpy.concatenate([
        weights_before_plateau,
        numpy.full(middle_length, 1.),
        weights_before_plateau[::-1]
    ])

    first_weight_matrix, second_weight_matrix = numpy.meshgrid(
        trapezoidal_weights, trapezoidal_weights
    )
    return first_weight_matrix * second_weight_matrix


def init_patch_metalocation_dict(patch_size_2pt5km_pixels,
                                 patch_overlap_size_2pt5km_pixels):
    """Initializes patch-metalocation dictionary.

    To understand what the "patch-metalocation dictionary" is, see documentation
    for `update_patch_metalocation_dict`.

    :param patch_size_2pt5km_pixels: See doc for
        `update_patch_metalocation_dict`.
    :param patch_overlap_size_2pt5km_pixels: Same.
    :return: patch_metalocation_dict: Same.
    """

    return {
        NUM_FULL_ROWS_KEY: len(nbm_utils.NBM_Y_COORDS_METRES),
        NUM_FULL_COLUMNS_KEY: len(nbm_utils.NBM_X_COORDS_METRES),
        NUM_PATCH_ROWS_KEY: patch_size_2pt5km_pixels,
        NUM_PATCH_COLUMNS_KEY: patch_size_2pt5km_pixels,
        PATCH_OVERLAP_SIZE_KEY: patch_overlap_size_2pt5km_pixels,
        PATCH_START_ROW_KEY: -1,
        PATCH_START_COLUMN_KEY: -1
    }


def update_patch_metalocation_dict(patch_metalocation_dict):
    """Updates patch-metalocation dictionary.

    This is fancy talk for "determines where the next patch will be, when
    applying a multi-patch neural net over the full grid".

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


def create_data(
        option_dict, patch_overlap_size_2pt5km_pixels, init_time_unix_sec,
        return_predictors_as_dict=False):
    """Creates NN-input data for inference.

    E = number of examples = number of patches in full NBM grid
    M = number of rows in full-resolution (2.5-km) grid
    N = number of columns in full-resolution (2.5-km) grid

    :param option_dict: See documentation for `data_generator`.
    :param patch_overlap_size_2pt5km_pixels: Same.
    :param init_time_unix_sec: Will return all patches for this initialization
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

    # Check input arguments.
    option_dict = nn_utils.check_generator_args(option_dict)

    # Read input arguments.
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
    sentinel_value = option_dict[nn_utils.SENTINEL_VALUE_KEY]
    patch_size_2pt5km_pixels = option_dict[nn_utils.PATCH_SIZE_KEY]
    patch_buffer_size_2pt5km_pixels = option_dict[
        nn_utils.PATCH_BUFFER_SIZE_KEY
    ]
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

    error_checking.assert_is_integer(patch_overlap_size_2pt5km_pixels)
    error_checking.assert_is_geq(patch_overlap_size_2pt5km_pixels, 16)

    nwp_model_names = list(nwp_model_to_dir_name.keys())
    nwp_model_names.sort()

    # Read normalization parameters.
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

    # Ensure that forecast-initialization time makes sense.
    init_times_unix_sec = nn_utils.find_relevant_init_times(
        first_time_by_period_unix_sec=first_init_times_unix_sec,
        last_time_by_period_unix_sec=last_init_times_unix_sec,
        nwp_model_names=nwp_model_names
    )

    # TODO(thunderhoser): This is a HACK.
    if use_recent_biases:
        good_indices = numpy.where(
            numpy.mod(init_times_unix_sec, 24 * HOURS_TO_SECONDS) ==
            18 * HOURS_TO_SECONDS
        )[0]
        init_times_unix_sec = init_times_unix_sec[good_indices]

    error_checking.assert_equals(len(init_times_unix_sec), 1)
    init_time_unix_sec = init_times_unix_sec[0]

    # Read time-invariant fields (land/sea mask, orographic height, etc.).
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

    # Read target fields (correct answers from URMA).
    if target_lag_times_hours is None:
        num_target_lag_times = 0
    else:
        num_target_lag_times = len(target_lag_times_hours)

    try:
        full_target_matrix = nn_utils.read_targets_one_example(
            init_time_unix_sec=init_time_unix_sec,
            target_lead_time_hours=target_lead_time_hours,
            target_field_names=target_field_names,
            target_dir_name=target_dir_name,
            target_norm_param_table_xarray=None,
            target_resid_norm_param_table_xarray=None,
            use_quantile_norm=False,
            patch_location_dict=None
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
        return None

    if full_target_matrix is None:
        return None

    # Read lagged-truth fields to use as predictors.
    # TODO(thunderhoser): Currently, lagged-truth fields come from URMA.  In the
    # future, lagged-truth fields will come from RTMA, since the RTMA is
    # available in real time and URMA is not.
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
                    patch_location_dict=None
                )
                for l in target_lag_times_hours
            ]

            full_predictor_matrix_lagged_targets = numpy.stack(
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
            return None
    else:
        full_predictor_matrix_lagged_targets = None

    if (
            num_target_lag_times > 0 and
            full_predictor_matrix_lagged_targets is None
    ):
        return None

    # Read residual baseline.  The "residual baseline" is the neural network's
    # default prediction.  In other words, the NN's task is to predict the
    # departure -- for every target field at the given lead time -- between the
    # residual baseline and the URMA truth field.  The residual baseline is
    # added to the NN's departure prediction, yielding a "full" prediction for
    # every target field.  This is done inside the NN architecture, so you don't
    # need to worry about it.
    if do_residual_prediction:
        try:
            full_baseline_matrix = nwp_input.read_residual_baseline_one_example(
                init_time_unix_sec=init_time_unix_sec,
                nwp_model_name=resid_baseline_model_name,
                nwp_lead_time_hours=resid_baseline_lead_time_hours,
                nwp_directory_name=resid_baseline_model_dir_name,
                target_field_names=target_field_names,
                patch_location_dict=None,
                predict_dewpoint_depression=True,
                predict_gust_excess=True
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
            return None
    else:
        full_baseline_matrix = None

    if do_residual_prediction and full_baseline_matrix is None:
        return None

    # Read NWP forecasts.  These are used as predictors.
    try:
        (
            full_predictor_matrix_2pt5km,
            full_predictor_matrix_10km,
            full_predictor_matrix_20km,
            full_predictor_matrix_40km,
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
            patch_location_dict=None
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
        return None

    if not found_any_predictors:
        return None
    if require_all_predictors and not found_all_predictors:
        return None

    # Read recent biases in NWP forecasts.  These are used as predictors.
    if use_recent_biases:
        try:
            (
                full_recent_bias_matrix_2pt5km,
                full_recent_bias_matrix_10km,
                full_recent_bias_matrix_20km,
                full_recent_bias_matrix_40km,
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
                patch_location_dict=None
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
            return None

        if not found_any_predictors:
            return None
        if require_all_predictors and not found_all_predictors:
            return None
    else:
        full_recent_bias_matrix_2pt5km = None
        full_recent_bias_matrix_10km = None
        full_recent_bias_matrix_20km = None
        full_recent_bias_matrix_40km = None

    # Initialize patch-metalocation dictionary.  This is explained more in
    # outside documentation.
    patch_metalocation_dict = init_patch_metalocation_dict(
        patch_size_2pt5km_pixels=patch_size_2pt5km_pixels,
        patch_overlap_size_2pt5km_pixels=patch_overlap_size_2pt5km_pixels
    )

    # Determine number of patches in the full NBM grid.
    pmld = patch_metalocation_dict
    num_patches = 0

    while True:
        pmld = update_patch_metalocation_dict(pmld)
        if pmld[PATCH_START_ROW_KEY] < 0:
            break

        num_patches += 1

    # Create binary mask for model evaluation.  This is explained more in
    # outside documentation.
    mask_matrix_for_loss = nn_utils.patch_buffer_to_mask(
        patch_size_2pt5km_pixels=patch_size_2pt5km_pixels,
        patch_buffer_size_2pt5km_pixels=patch_buffer_size_2pt5km_pixels
    )
    mask_matrix_for_loss = numpy.repeat(
        numpy.expand_dims(mask_matrix_for_loss, axis=0),
        repeats=num_patches,
        axis=0
    )
    mask_matrix_for_loss = numpy.expand_dims(mask_matrix_for_loss, axis=-1)

    # Initialize output arrays.
    dummy_patch_location_dict = misc_utils.determine_patch_locations(
        patch_size_2pt5km_pixels=patch_size_2pt5km_pixels
    )

    matrix_dict = nn_utils.init_matrices_1batch(
        nwp_model_names=nwp_model_names,
        nwp_model_to_field_names=nwp_model_to_field_names,
        num_nwp_lead_times=len(nwp_lead_times_hours),
        target_field_names=target_field_names,
        num_recent_bias_times=num_recent_bias_times,
        num_target_lag_times=num_target_lag_times,
        num_examples_per_batch=num_patches,
        patch_location_dict=dummy_patch_location_dict,
        do_residual_prediction=do_residual_prediction
    )
    predictor_matrix_2pt5km = matrix_dict[nn_utils.PREDICTOR_MATRIX_2PT5KM_KEY]
    predictor_matrix_10km = matrix_dict[nn_utils.PREDICTOR_MATRIX_10KM_KEY]
    predictor_matrix_20km = matrix_dict[nn_utils.PREDICTOR_MATRIX_20KM_KEY]
    predictor_matrix_40km = matrix_dict[nn_utils.PREDICTOR_MATRIX_40KM_KEY]
    recent_bias_matrix_2pt5km = matrix_dict[
        nn_utils.RECENT_BIAS_MATRIX_2PT5KM_KEY
    ]
    recent_bias_matrix_10km = matrix_dict[nn_utils.RECENT_BIAS_MATRIX_10KM_KEY]
    recent_bias_matrix_20km = matrix_dict[nn_utils.RECENT_BIAS_MATRIX_20KM_KEY]
    recent_bias_matrix_40km = matrix_dict[nn_utils.RECENT_BIAS_MATRIX_40KM_KEY]
    predictor_matrix_resid_baseline = matrix_dict[
        nn_utils.PREDICTOR_MATRIX_BASELINE_KEY
    ]
    predictor_matrix_lagged_targets = matrix_dict[
        nn_utils.PREDICTOR_MATRIX_LAGTGT_KEY
    ]
    target_matrix = matrix_dict[nn_utils.TARGET_MATRIX_KEY]

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

    # Populate output arrays.  This involves extracting data from "full" arrays
    # (containing the full NBM grid) and putting the same data into "patch"
    # arrays, where the patch size in the latter arrays is the same patch size
    # used in training.
    patch_metalocation_dict = init_patch_metalocation_dict(
        patch_size_2pt5km_pixels=patch_size_2pt5km_pixels,
        patch_overlap_size_2pt5km_pixels=patch_overlap_size_2pt5km_pixels
    )

    for i in range(num_patches):
        patch_metalocation_dict = update_patch_metalocation_dict(
            patch_metalocation_dict
        )
        patch_location_dict = misc_utils.determine_patch_locations(
            patch_size_2pt5km_pixels=patch_size_2pt5km_pixels,
            start_row_2pt5km=patch_metalocation_dict[PATCH_START_ROW_KEY],
            start_column_2pt5km=patch_metalocation_dict[PATCH_START_COLUMN_KEY]
        )
        pld = patch_location_dict

        j_start = pld[misc_utils.ROW_LIMITS_2PT5KM_KEY][0]
        j_end = pld[misc_utils.ROW_LIMITS_2PT5KM_KEY][1] + 1
        k_start = pld[misc_utils.COLUMN_LIMITS_2PT5KM_KEY][0]
        k_end = pld[misc_utils.COLUMN_LIMITS_2PT5KM_KEY][1] + 1

        target_matrix[i, ...] = (
            full_target_matrix[j_start:j_end, k_start:k_end, ...]
        )
        latitude_matrix_deg_n[i, ...] = (
            full_latitude_matrix_deg_n[j_start:j_end, k_start:k_end]
        )
        longitude_matrix_deg_e[i, ...] = (
            full_longitude_matrix_deg_e[j_start:j_end, k_start:k_end]
        )

        if do_residual_prediction:
            predictor_matrix_resid_baseline[i, ...] = (
                full_baseline_matrix[j_start:j_end, k_start:k_end, ...]
            )

        if predictor_matrix_lagged_targets is not None:
            predictor_matrix_lagged_targets[i, ...] = (
                full_predictor_matrix_lagged_targets[j_start:j_end, k_start:k_end, ...]
            )

        if predictor_matrix_2pt5km is not None:
            predictor_matrix_2pt5km[i, ...] = (
                full_predictor_matrix_2pt5km[j_start:j_end, k_start:k_end, ...]
            )

        if recent_bias_matrix_2pt5km is not None:
            recent_bias_matrix_2pt5km[i, ...] = (
                full_recent_bias_matrix_2pt5km[j_start:j_end, k_start:k_end, ...]
            )

        if nbm_constant_matrix is not None:
            nbm_constant_matrix[i, ...] = (
                full_nbm_constant_matrix[j_start:j_end, k_start:k_end, ...]
            )

        j_start = pld[misc_utils.ROW_LIMITS_10KM_KEY][0]
        j_end = pld[misc_utils.ROW_LIMITS_10KM_KEY][1] + 1
        k_start = pld[misc_utils.COLUMN_LIMITS_10KM_KEY][0]
        k_end = pld[misc_utils.COLUMN_LIMITS_10KM_KEY][1] + 1

        if predictor_matrix_10km is not None:
            predictor_matrix_10km[i, ...] = (
                full_predictor_matrix_10km[j_start:j_end, k_start:k_end, ...]
            )

        if recent_bias_matrix_10km is not None:
            recent_bias_matrix_10km[i, ...] = (
                full_recent_bias_matrix_10km[j_start:j_end, k_start:k_end, ...]
            )

        j_start = pld[misc_utils.ROW_LIMITS_20KM_KEY][0]
        j_end = pld[misc_utils.ROW_LIMITS_20KM_KEY][1] + 1
        k_start = pld[misc_utils.COLUMN_LIMITS_20KM_KEY][0]
        k_end = pld[misc_utils.COLUMN_LIMITS_20KM_KEY][1] + 1

        if predictor_matrix_20km is not None:
            predictor_matrix_20km[i, ...] = (
                full_predictor_matrix_20km[j_start:j_end, k_start:k_end, ...]
            )

        if recent_bias_matrix_20km is not None:
            recent_bias_matrix_20km[i, ...] = (
                full_recent_bias_matrix_20km[j_start:j_end, k_start:k_end, ...]
            )

        j_start = pld[misc_utils.ROW_LIMITS_40KM_KEY][0]
        j_end = pld[misc_utils.ROW_LIMITS_40KM_KEY][1] + 1
        k_start = pld[misc_utils.COLUMN_LIMITS_40KM_KEY][0]
        k_end = pld[misc_utils.COLUMN_LIMITS_40KM_KEY][1] + 1

        if predictor_matrix_40km is not None:
            predictor_matrix_40km[i, ...] = (
                full_predictor_matrix_40km[j_start:j_end, k_start:k_end, ...]
            )

        if recent_bias_matrix_40km is not None:
            recent_bias_matrix_40km[i, ...] = (
                full_recent_bias_matrix_40km[j_start:j_end, k_start:k_end, ...]
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

    return {
        PREDICTOR_MATRICES_KEY: list(predictor_matrices),
        TARGET_MATRIX_KEY: target_matrix,
        INIT_TIMES_KEY: numpy.full(num_patches, init_time_unix_sec),
        LATITUDE_MATRIX_KEY: latitude_matrix_deg_n,
        LONGITUDE_MATRIX_KEY: longitude_matrix_deg_e
    }


def data_generator_from_example_files(
        example_dir_name, first_init_times_unix_sec, last_init_times_unix_sec,
        num_examples_per_batch, patch_size_2pt5km_pixels,
        patch_buffer_size_2pt5km_pixels, patch_overlap_size_2pt5km_pixels,
        return_predictors_as_dict=False):
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
    :param patch_size_2pt5km_pixels: Patch size, in units of 2.5-km pixels.  For
        example, if patch_size_2pt5km_pixels = 448, then grid dimensions at the
        finest resolution (2.5 km) are 448 x 448.
    :param patch_buffer_size_2pt5km_pixels: Buffer between the outer domain
        (used for predictors) and the inner domain (used to penalize predictions
        in loss function).  This must be a non-negative integer.
    :param patch_overlap_size_2pt5km_pixels: Overlap between adjacent patches,
        in terms of 2.5-km pixels.
        
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

    error_checking.assert_is_integer(patch_size_2pt5km_pixels)
    error_checking.assert_is_greater(patch_size_2pt5km_pixels, 0)
    error_checking.assert_is_integer(patch_buffer_size_2pt5km_pixels)
    error_checking.assert_is_geq(patch_buffer_size_2pt5km_pixels, 0)
    error_checking.assert_is_less_than(
        patch_buffer_size_2pt5km_pixels, patch_size_2pt5km_pixels // 2
    )
    error_checking.assert_is_integer(patch_overlap_size_2pt5km_pixels)
    error_checking.assert_is_geq(patch_overlap_size_2pt5km_pixels, 16)

    # Do actual stuff.
    init_times_unix_sec = nn_utils.find_relevant_init_times(
        first_time_by_period_unix_sec=first_init_times_unix_sec,
        last_time_by_period_unix_sec=last_init_times_unix_sec,
        nwp_model_names=[nwp_model_utils.HRRR_MODEL_NAME]
    )
    numpy.random.shuffle(init_times_unix_sec)
    init_time_index = 0

    patch_metalocation_dict = init_patch_metalocation_dict(
        patch_size_2pt5km_pixels=patch_size_2pt5km_pixels,
        patch_overlap_size_2pt5km_pixels=patch_overlap_size_2pt5km_pixels
    )
    full_predictor_matrices = None
    full_target_matrix = None
    num_rows_in_full_grid = len(nbm_utils.NBM_Y_COORDS_METRES)

    while True:
        num_examples_in_memory = 0
        predictor_matrices = None
        target_matrix = None

        while num_examples_in_memory < num_examples_per_batch:
            patch_metalocation_dict = update_patch_metalocation_dict(
                patch_metalocation_dict
            )

            if patch_metalocation_dict[PATCH_START_ROW_KEY] < 0:
                full_predictor_matrices = None
                full_target_matrix = None

                init_time_index, init_times_unix_sec = (
                    nn_utils.increment_init_time(
                        current_index=init_time_index,
                        init_times_unix_sec=init_times_unix_sec
                    )
                )
                continue

            if full_predictor_matrices is None:
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
                full_predictor_matrices, full_target_matrix = (
                    example_io.read_file(example_file_name)
                )

            patch_location_dict = misc_utils.determine_patch_locations(
                patch_size_2pt5km_pixels=patch_size_2pt5km_pixels,
                start_row_2pt5km=patch_metalocation_dict[PATCH_START_ROW_KEY],
                start_column_2pt5km=
                patch_metalocation_dict[PATCH_START_COLUMN_KEY]
            )
            pld = patch_location_dict

            if predictor_matrices is None:
                these_dim = (
                    num_examples_per_batch, patch_size_2pt5km_pixels,
                    patch_size_2pt5km_pixels
                )
                these_dim = these_dim + target_matrix.shape[3:]
                target_matrix = numpy.full(these_dim, numpy.nan)

                predictor_matrices = (
                    [numpy.array([])] * len(full_predictor_matrices)
                )

                for m in range(len(full_predictor_matrices)):
                    if (
                            full_predictor_matrices[m].shape[1] ==
                            num_rows_in_full_grid
                    ):
                        these_dim = (
                            num_examples_per_batch, patch_size_2pt5km_pixels,
                            patch_size_2pt5km_pixels
                        )

                    elif numpy.isclose(
                            full_predictor_matrices[m].shape[1],
                            num_rows_in_full_grid // 4, atol=1
                    ):
                        these_dim = (
                            num_examples_per_batch, patch_size_2pt5km_pixels // 4,
                            patch_size_2pt5km_pixels // 4
                        )

                    elif numpy.isclose(
                            full_predictor_matrices[m].shape[1],
                            num_rows_in_full_grid // 8, atol=1
                    ):
                        these_dim = (
                            num_examples_per_batch, patch_size_2pt5km_pixels // 8,
                            patch_size_2pt5km_pixels // 8
                        )

                    elif numpy.isclose(
                            full_predictor_matrices[m].shape[1],
                            num_rows_in_full_grid // 16, atol=1
                    ):
                        these_dim = (
                            num_examples_per_batch, patch_size_2pt5km_pixels // 16,
                            patch_size_2pt5km_pixels // 16
                        )

                    these_dim = these_dim + full_predictor_matrices[m].shape[3:]
                    predictor_matrices[m] = numpy.full(these_dim, numpy.nan)

            j_start = pld[misc_utils.ROW_LIMITS_2PT5KM_KEY][0]
            j_end = pld[misc_utils.ROW_LIMITS_2PT5KM_KEY][1] + 1
            k_start = pld[misc_utils.COLUMN_LIMITS_2PT5KM_KEY][0]
            k_end = pld[misc_utils.COLUMN_LIMITS_2PT5KM_KEY][1] + 1
            i = num_examples_in_memory + 0

            target_matrix[i, ...] = (
                full_target_matrix[0, j_start:j_end, k_start:k_end, ...]
            )

            for m in range(len(predictor_matrices)):
                if predictor_matrices[m].shape[1] == patch_size_2pt5km_pixels:
                    j_start = pld[misc_utils.ROW_LIMITS_2PT5KM_KEY][0]
                    j_end = pld[misc_utils.ROW_LIMITS_2PT5KM_KEY][1] + 1
                    k_start = pld[misc_utils.COLUMN_LIMITS_2PT5KM_KEY][0]
                    k_end = pld[misc_utils.COLUMN_LIMITS_2PT5KM_KEY][1] + 1
                elif (
                        predictor_matrices[m].shape[1] ==
                        patch_size_2pt5km_pixels // 4
                ):
                    j_start = pld[misc_utils.ROW_LIMITS_10KM_KEY][0]
                    j_end = pld[misc_utils.ROW_LIMITS_10KM_KEY][1] + 1
                    k_start = pld[misc_utils.COLUMN_LIMITS_10KM_KEY][0]
                    k_end = pld[misc_utils.COLUMN_LIMITS_10KM_KEY][1] + 1
                elif (
                        predictor_matrices[m].shape[1] ==
                        patch_size_2pt5km_pixels // 8
                ):
                    j_start = pld[misc_utils.ROW_LIMITS_20KM_KEY][0]
                    j_end = pld[misc_utils.ROW_LIMITS_20KM_KEY][1] + 1
                    k_start = pld[misc_utils.COLUMN_LIMITS_20KM_KEY][0]
                    k_end = pld[misc_utils.COLUMN_LIMITS_20KM_KEY][1] + 1
                elif (
                        predictor_matrices[m].shape[1] ==
                        patch_size_2pt5km_pixels // 16
                ):
                    j_start = pld[misc_utils.ROW_LIMITS_40KM_KEY][0]
                    j_end = pld[misc_utils.ROW_LIMITS_40KM_KEY][1] + 1
                    k_start = pld[misc_utils.COLUMN_LIMITS_40KM_KEY][0]
                    k_end = pld[misc_utils.COLUMN_LIMITS_40KM_KEY][1] + 1

                predictor_matrices[m][i, ...] = (
                    full_predictor_matrices[m][0, j_start:j_end, k_start:k_end, ...]
                )

            if numpy.any(numpy.isnan(target_matrix[i, ...])):
                continue

            num_examples_in_memory += 1

        yield predictor_matrices, target_matrix


def data_generator(
        option_dict, patch_overlap_size_2pt5km_pixels,
        return_predictors_as_dict=False):
    """Generates data for multi-patch training.

    E = number of examples per batch = "batch size"
    M = number of rows in 2.5-km-resolution patch
    N = number of columns in 2.5-km-resolution patch
    P = number of NWP fields (predictor variables) at 2.5-km resolution
    C = number of constant fields (at 2.5-km resolution)
    L = number of NWP lead times
    l = number of lag times for target fields used in predictors
    F = number of target fields
    B = number of lag times for recent NWP bias

    m = number of rows in 10-km-resolution patch
    n = number of columns in 10-km-resolution patch
    p = number of NWP fields at 10-km resolution
    mm = number of rows in 20-km-resolution patch
    nn = number of columns in 20-km-resolution patch
    pp = number of NWP fields at 20-km resolution
    mmm = number of rows in 40-km-resolution patch
    nnn = number of columns in 40-km-resolution patch
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
        dimensions at the finest resolution (2.5 km) are 448 x 448.
    option_dict["patch_buffer_size_2pt5km_pixels"]:
        Buffer between the outer domain (used for predictors) and the inner
        domain (used to penalize predictions in loss function).  This must be a
        non-negative integer.
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

    :param patch_overlap_size_2pt5km_pixels: Overlap between adjacent patches,
        in terms of 2.5-km pixels.
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

    # TODO(thunderhoser): This is a HACK.
    if use_recent_biases:
        good_indices = numpy.where(
            numpy.mod(init_times_unix_sec, 24 * HOURS_TO_SECONDS) ==
            18 * HOURS_TO_SECONDS
        )[0]
        init_times_unix_sec = init_times_unix_sec[good_indices]

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

    patch_metalocation_dict = init_patch_metalocation_dict(
        patch_size_2pt5km_pixels=patch_size_2pt5km_pixels,
        patch_overlap_size_2pt5km_pixels=patch_overlap_size_2pt5km_pixels
    )

    # TODO(thunderhoser): Clean this up with a dictionary or something.
    full_target_matrix = full_baseline_matrix = \
        full_predictor_matrix_2pt5km = full_predictor_matrix_10km = \
        full_predictor_matrix_20km = full_predictor_matrix_40km = \
        full_recent_bias_matrix_2pt5km = full_recent_bias_matrix_10km = \
        full_recent_bias_matrix_20km = full_recent_bias_matrix_40km = \
        full_predictor_matrix_lagged_targets = None

    num_target_fields = len(target_field_names)
    if target_lag_times_hours is None:
        num_target_lag_times = 0
    else:
        num_target_lag_times = len(target_lag_times_hours)

    while True:
        dummy_patch_location_dict = misc_utils.determine_patch_locations(
            patch_size_2pt5km_pixels=patch_size_2pt5km_pixels
        )

        matrix_dict = nn_utils.init_matrices_1batch(
            nwp_model_names=nwp_model_names,
            nwp_model_to_field_names=nwp_model_to_field_names,
            num_nwp_lead_times=len(nwp_lead_times_hours),
            target_field_names=target_field_names,
            num_target_lag_times=num_target_lag_times,
            num_recent_bias_times=num_recent_bias_times,
            num_examples_per_batch=num_examples_per_batch,
            patch_location_dict=dummy_patch_location_dict,
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
            patch_metalocation_dict = update_patch_metalocation_dict(
                patch_metalocation_dict
            )

            if patch_metalocation_dict[PATCH_START_ROW_KEY] < 0:
                full_target_matrix = full_baseline_matrix = \
                    full_predictor_matrix_2pt5km = full_predictor_matrix_10km = \
                    full_predictor_matrix_20km = full_predictor_matrix_40km = \
                    full_recent_bias_matrix_2pt5km = full_recent_bias_matrix_10km = \
                    full_recent_bias_matrix_20km = full_recent_bias_matrix_40km = \
                    full_predictor_matrix_lagged_targets = None

                init_time_index, init_times_unix_sec = (
                    nn_utils.increment_init_time(
                        current_index=init_time_index,
                        init_times_unix_sec=init_times_unix_sec
                    )
                )
                continue

            try:
                if full_target_matrix is None:
                    full_target_matrix = nn_utils.read_targets_one_example(
                        init_time_unix_sec=init_times_unix_sec[init_time_index],
                        target_lead_time_hours=target_lead_time_hours,
                        target_field_names=target_field_names,
                        target_dir_name=target_dir_name,
                        target_norm_param_table_xarray=None,
                        target_resid_norm_param_table_xarray=None,
                        use_quantile_norm=False,
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

            if full_target_matrix is None:
                full_target_matrix = full_baseline_matrix = \
                    full_predictor_matrix_2pt5km = full_predictor_matrix_10km = \
                    full_predictor_matrix_20km = full_predictor_matrix_40km = \
                    full_recent_bias_matrix_2pt5km = full_recent_bias_matrix_10km = \
                    full_recent_bias_matrix_20km = full_recent_bias_matrix_40km = \
                    full_predictor_matrix_lagged_targets = None

                init_time_index, init_times_unix_sec = (
                    nn_utils.increment_init_time(
                        current_index=init_time_index,
                        init_times_unix_sec=init_times_unix_sec
                    )
                )
                continue

            try:
                if (
                        num_target_lag_times > 0 and
                        full_predictor_matrix_lagged_targets is None
                ):
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
                            patch_location_dict=None
                        )
                        for l in target_lag_times_hours
                    ]

                    full_predictor_matrix_lagged_targets = numpy.stack(
                        these_matrices, axis=-2
                    )
            except:
                warning_string = (
                    'POTENTIAL ERROR: Could not read lagged targets for init '
                    'time {0:s}.  Something went wrong in '
                    '`_read_targets_one_example`.'
                ).format(
                    time_conversion.unix_sec_to_string(
                        init_times_unix_sec[init_time_index], '%Y-%m-%d-%H'
                    )
                )

                warnings.warn(warning_string)

            if (
                    num_target_lag_times > 0 and
                    full_predictor_matrix_lagged_targets is None
            ):
                full_target_matrix = full_baseline_matrix = \
                    full_predictor_matrix_2pt5km = full_predictor_matrix_10km = \
                    full_predictor_matrix_20km = full_predictor_matrix_40km = \
                    full_recent_bias_matrix_2pt5km = full_recent_bias_matrix_10km = \
                    full_recent_bias_matrix_20km = full_recent_bias_matrix_40km = \
                    full_predictor_matrix_lagged_targets = None

                init_time_index, init_times_unix_sec = (
                    nn_utils.increment_init_time(
                        current_index=init_time_index,
                        init_times_unix_sec=init_times_unix_sec
                    )
                )
                continue

            need_baseline = (
                    do_residual_prediction or compare_to_baseline_in_loss
            )

            try:
                if need_baseline and full_baseline_matrix is None:
                    full_baseline_matrix = (
                        nwp_input.read_residual_baseline_one_example(
                            init_time_unix_sec=
                            init_times_unix_sec[init_time_index],
                            nwp_model_name=resid_baseline_model_name,
                            nwp_lead_time_hours=resid_baseline_lead_time_hours,
                            nwp_directory_name=resid_baseline_model_dir_name,
                            target_field_names=target_field_names,
                            patch_location_dict=None,
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
                                    init_time_unix_sec=
                                    init_times_unix_sec[init_time_index],
                                    nwp_model_name=resid_baseline_model_name,
                                    nwp_lead_time_hours=
                                    resid_baseline_lead_time_hours,
                                    nwp_directory_name=
                                    resid_baseline_model_dir_name,
                                    target_field_names=target_field_names,
                                    patch_location_dict=None,
                                    predict_dewpoint_depression=False,
                                    predict_gust_excess=False
                                )
                            )

                            full_target_matrix = numpy.concatenate(
                                [full_target_matrix, raw_baseline_matrix],
                                axis=-1
                            )
                        else:
                            full_target_matrix = numpy.concatenate(
                                [full_target_matrix, full_baseline_matrix],
                                axis=-1
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

            if need_baseline and full_baseline_matrix is None:
                full_target_matrix = full_baseline_matrix = \
                    full_predictor_matrix_2pt5km = full_predictor_matrix_10km = \
                    full_predictor_matrix_20km = full_predictor_matrix_40km = \
                    full_recent_bias_matrix_2pt5km = full_recent_bias_matrix_10km = \
                    full_recent_bias_matrix_20km = full_recent_bias_matrix_40km = \
                    full_predictor_matrix_lagged_targets = None

                init_time_index, init_times_unix_sec = (
                    nn_utils.increment_init_time(
                        current_index=init_time_index,
                        init_times_unix_sec=init_times_unix_sec
                    )
                )
                continue

            try:
                if full_predictor_matrix_2pt5km is None:
                    (
                        full_predictor_matrix_2pt5km,
                        full_predictor_matrix_10km,
                        full_predictor_matrix_20km,
                        full_predictor_matrix_40km,
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
                        backup_nwp_model_name=backup_nwp_model_name,
                        backup_nwp_directory_name=backup_nwp_directory_name,
                        patch_location_dict=None
                    )
                else:
                    found_any_predictors = True
                    found_all_predictors = True
            except:
                warning_string = (
                    'POTENTIAL ERROR: Could not read predictors for init time '
                    '{0:s}.  Something went wrong in '
                    '`nwp_input.read_predictors_one_example`.'
                ).format(
                    time_conversion.unix_sec_to_string(
                        init_times_unix_sec[init_time_index], '%Y-%m-%d-%H'
                    )
                )

                warnings.warn(warning_string)
                found_any_predictors = False
                found_all_predictors = False

            if not found_any_predictors:
                full_target_matrix = full_baseline_matrix = \
                    full_predictor_matrix_2pt5km = full_predictor_matrix_10km = \
                    full_predictor_matrix_20km = full_predictor_matrix_40km = \
                    full_recent_bias_matrix_2pt5km = full_recent_bias_matrix_10km = \
                    full_recent_bias_matrix_20km = full_recent_bias_matrix_40km = \
                    full_predictor_matrix_lagged_targets = None

                init_time_index, init_times_unix_sec = (
                    nn_utils.increment_init_time(
                        current_index=init_time_index,
                        init_times_unix_sec=init_times_unix_sec
                    )
                )
                continue

            if require_all_predictors and not found_all_predictors:
                full_target_matrix = full_baseline_matrix = \
                    full_predictor_matrix_2pt5km = full_predictor_matrix_10km = \
                    full_predictor_matrix_20km = full_predictor_matrix_40km = \
                    full_recent_bias_matrix_2pt5km = full_recent_bias_matrix_10km = \
                    full_recent_bias_matrix_20km = full_recent_bias_matrix_40km = \
                    full_predictor_matrix_lagged_targets = None

                init_time_index, init_times_unix_sec = (
                    nn_utils.increment_init_time(
                        current_index=init_time_index,
                        init_times_unix_sec=init_times_unix_sec
                    )
                )
                continue

            try:
                if use_recent_biases and full_recent_bias_matrix_2pt5km is None:
                    (
                        full_recent_bias_matrix_2pt5km,
                        full_recent_bias_matrix_10km,
                        full_recent_bias_matrix_20km,
                        full_recent_bias_matrix_40km,
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
                        patch_location_dict=None
                    )
                else:
                    found_any_predictors = True
                    found_all_predictors = True
            except:
                warning_string = (
                    'POTENTIAL ERROR: Could not read recent biases for init '
                    'time {0:s}.  Something went wrong in '
                    '`nwp_input.read_recent_biases_one_example`.'
                ).format(
                    time_conversion.unix_sec_to_string(
                        init_times_unix_sec[init_time_index], '%Y-%m-%d-%H'
                    )
                )

                warnings.warn(warning_string)
                found_any_predictors = False
                found_all_predictors = False

            # TODO(thunderhoser): I don't think this condition needs to be
            # `use_recent_biases and not found_any_predictors`?
            if not found_any_predictors:
                full_target_matrix = full_baseline_matrix = \
                    full_predictor_matrix_2pt5km = full_predictor_matrix_10km = \
                    full_predictor_matrix_20km = full_predictor_matrix_40km = \
                    full_recent_bias_matrix_2pt5km = full_recent_bias_matrix_10km = \
                    full_recent_bias_matrix_20km = full_recent_bias_matrix_40km = \
                    full_predictor_matrix_lagged_targets = None

                init_time_index, init_times_unix_sec = (
                    nn_utils.increment_init_time(
                        current_index=init_time_index,
                        init_times_unix_sec=init_times_unix_sec
                    )
                )
                continue

            if require_all_predictors and not found_all_predictors:
                full_target_matrix = full_baseline_matrix = \
                    full_predictor_matrix_2pt5km = full_predictor_matrix_10km = \
                    full_predictor_matrix_20km = full_predictor_matrix_40km = \
                    full_recent_bias_matrix_2pt5km = full_recent_bias_matrix_10km = \
                    full_recent_bias_matrix_20km = full_recent_bias_matrix_40km = \
                    full_predictor_matrix_lagged_targets = None

                init_time_index, init_times_unix_sec = (
                    nn_utils.increment_init_time(
                        current_index=init_time_index,
                        init_times_unix_sec=init_times_unix_sec
                    )
                )
                continue

            patch_location_dict = misc_utils.determine_patch_locations(
                patch_size_2pt5km_pixels=patch_size_2pt5km_pixels,
                start_row_2pt5km=patch_metalocation_dict[PATCH_START_ROW_KEY],
                start_column_2pt5km=
                patch_metalocation_dict[PATCH_START_COLUMN_KEY]
            )
            pld = patch_location_dict

            j_start = pld[misc_utils.ROW_LIMITS_2PT5KM_KEY][0]
            j_end = pld[misc_utils.ROW_LIMITS_2PT5KM_KEY][1] + 1
            k_start = pld[misc_utils.COLUMN_LIMITS_2PT5KM_KEY][0]
            k_end = pld[misc_utils.COLUMN_LIMITS_2PT5KM_KEY][1] + 1
            i = num_examples_in_memory + 0

            target_matrix[i, ...] = (
                full_target_matrix[j_start:j_end, k_start:k_end, ...]
            )

            # TODO(thunderhoser): This can happen along edges of grid for
            # earlier URMA data.
            # if numpy.any(numpy.isnan(
            #         target_matrix[i, ..., :num_target_fields]
            # )):
            #     continue

            # TODO(thunderhoser): This can also happen along edges of grid for
            # residual baseline, if residual baseline is high-res ensemble.
            if numpy.any(numpy.isnan(
                    target_matrix[i, num_target_fields:(2 * num_target_fields)]
            )):
                print('Residual baseline contains NaN values ({0:.6f}%).'.format(
                    100 * numpy.mean(numpy.isnan(
                        target_matrix[i, num_target_fields:(2 * num_target_fields)]
                    ))
                ))
            else:
                print('Residual baseline does NOT contain NaN values.')

            if numpy.any(numpy.isnan(target_matrix[i, ...])):
                continue

            if do_residual_prediction:
                predictor_matrix_resid_baseline[i, ...] = (
                    full_baseline_matrix[j_start:j_end, k_start:k_end, ...]
                )

            if predictor_matrix_lagged_targets is not None:
                predictor_matrix_lagged_targets[i, ...] = (
                    full_predictor_matrix_lagged_targets[j_start:j_end, k_start:k_end, ...]
                )

            if predictor_matrix_2pt5km is not None:
                predictor_matrix_2pt5km[i, ...] = (
                    full_predictor_matrix_2pt5km[j_start:j_end, k_start:k_end, ...]
                )

            if nbm_constant_matrix is not None:
                nbm_constant_matrix[i, ...] = (
                    full_nbm_constant_matrix[j_start:j_end, k_start:k_end, ...]
                )

            if recent_bias_matrix_2pt5km is not None:
                recent_bias_matrix_2pt5km[i, ...] = (
                    full_recent_bias_matrix_2pt5km[j_start:j_end, k_start:k_end, ...]
                )

            j_start = pld[misc_utils.ROW_LIMITS_10KM_KEY][0]
            j_end = pld[misc_utils.ROW_LIMITS_10KM_KEY][1] + 1
            k_start = pld[misc_utils.COLUMN_LIMITS_10KM_KEY][0]
            k_end = pld[misc_utils.COLUMN_LIMITS_10KM_KEY][1] + 1

            if predictor_matrix_10km is not None:
                predictor_matrix_10km[i, ...] = (
                    full_predictor_matrix_10km[j_start:j_end, k_start:k_end, ...]
                )

            if recent_bias_matrix_10km is not None:
                recent_bias_matrix_10km[i, ...] = (
                    full_recent_bias_matrix_10km[j_start:j_end, k_start:k_end, ...]
                )

            j_start = pld[misc_utils.ROW_LIMITS_20KM_KEY][0]
            j_end = pld[misc_utils.ROW_LIMITS_20KM_KEY][1] + 1
            k_start = pld[misc_utils.COLUMN_LIMITS_20KM_KEY][0]
            k_end = pld[misc_utils.COLUMN_LIMITS_20KM_KEY][1] + 1

            if predictor_matrix_20km is not None:
                predictor_matrix_20km[i, ...] = (
                    full_predictor_matrix_20km[j_start:j_end, k_start:k_end, ...]
                )

            if recent_bias_matrix_20km is not None:
                recent_bias_matrix_20km[i, ...] = (
                    full_recent_bias_matrix_20km[j_start:j_end, k_start:k_end, ...]
                )

            j_start = pld[misc_utils.ROW_LIMITS_40KM_KEY][0]
            j_end = pld[misc_utils.ROW_LIMITS_40KM_KEY][1] + 1
            k_start = pld[misc_utils.COLUMN_LIMITS_40KM_KEY][0]
            k_end = pld[misc_utils.COLUMN_LIMITS_40KM_KEY][1] + 1

            if predictor_matrix_40km is not None:
                predictor_matrix_40km[i, ...] = (
                    full_predictor_matrix_40km[j_start:j_end, k_start:k_end, ...]
                )

            if recent_bias_matrix_40km is not None:
                recent_bias_matrix_40km[i, ...] = (
                    full_recent_bias_matrix_40km[j_start:j_end, k_start:k_end, ...]
                )

            num_examples_in_memory += 1

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


def data_generator_for_u_net(option_dict, patch_overlap_size_2pt5km_pixels):
    """Data-generator for multi-patch training of simple U-net.

    E = number of examples per batch = "batch size"
    M = number of rows in NBM grid (2.5-km target grid)
    N = number of columns in NBM grid (2.5-km target grid)
    P = number of NWP fields (predictor variables) at 2.5-km resolution
    F = number of target fields

    :param option_dict: Dictionary with the following keys.
    option_dict["first_init_time_unix_sec"]: Start of training period, i.e.,
        first forecast-initialization time.
    option_dict["last_init_time_unix_sec"]: End of training period, i.e.,
        last forecast-initialization time.
    option_dict["nwp_lead_time_hours"]: Lead time of NWP forecasts used as
        predictors.
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
    option_dict["backup_nwp_model_name"]: Name of backup model, used to fill
        missing data.
    option_dict["backup_nwp_directory_name"]: Directory for backup model.  Files
        therein will be found by `interp_nwp_model_io.find_file`.
    option_dict["target_lead_time_hours"]: Lead time for target fields.
    option_dict["target_field_names"]: length-F list with names of target
        fields.  Each must be accepted by `urma_utils.check_field_name`.
    option_dict["target_dir_name"]: Path to directory with target fields (i.e.,
        URMA data).  Files within this directory will be found by
        `urma_io.find_file` and read by `urma_io.read_file`.
    option_dict["compare_to_baseline_in_loss"]: Boolean flag.  If True, the loss
        function involves comparing to the residual baseline.  In other words,
        the loss function involves a skill score, except with the residual
        baseline instead of climo.
    option_dict["num_examples_per_batch"]: Number of data examples per batch,
        usually just called 'batch size'.
    option_dict["sentinel_value"]: All NaN will be replaced with this value.
    option_dict["patch_size_2pt5km_pixels"]: Patch size, in units of 2.5-km
        pixels.  For example, if patch_size_2pt5km_pixels = 448, then grid
        dimensions at the finest resolution (2.5 km) are 448 x 448.
    option_dict["patch_buffer_size_2pt5km_pixels"]: Buffer between the outer
        domain (used for predictors) and the inner domain (used to penalize
        predictions in loss function).  This must be a non-negative integer.
    option_dict["resid_baseline_model_name"]: Name of NWP model used to
        generate residual baseline fields.  If
        compare_to_baseline_in_loss == False, make this argument None.
    option_dict["resid_baseline_lead_time_hours"]: Lead time used to generate
        residual baseline fields.  If compare_to_baseline_in_loss == False, make
        this argument None.
    option_dict["resid_baseline_model_dir_name"]: Directory path for residual
        baseline fields.  Within this directory, relevant files will be found by
        `interp_nwp_model_io.find_file`.

    :param patch_overlap_size_2pt5km_pixels: Overlap between adjacent patches,
        measured in number of pixels on the 2.5-km NBM grid.
    :return: predictor_matrix: E-by-M-by-N-by-P numpy array of predictors, all
        NWP forecasts at 2.5-km resolution.
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

    # Read input arguments.
    option_dict = nn_utils.check_u_net_generator_args(option_dict)
    first_init_time_unix_sec = option_dict['first_init_time_unix_sec']
    last_init_time_unix_sec = option_dict['last_init_time_unix_sec']
    nwp_lead_time_hours = option_dict['nwp_lead_time_hours']
    nwp_model_to_dir_name = option_dict[nn_utils.NWP_MODEL_TO_DIR_KEY]
    nwp_model_to_field_names = option_dict[nn_utils.NWP_MODEL_TO_FIELDS_KEY]
    nwp_normalization_file_name = option_dict[nn_utils.NWP_NORM_FILE_KEY]

    backup_nwp_model_name = option_dict[nn_utils.BACKUP_NWP_MODEL_KEY]
    backup_nwp_directory_name = option_dict[nn_utils.BACKUP_NWP_DIR_KEY]
    target_lead_time_hours = option_dict[nn_utils.TARGET_LEAD_TIME_KEY]
    target_field_names = option_dict[nn_utils.TARGET_FIELDS_KEY]
    target_dir_name = option_dict[nn_utils.TARGET_DIR_KEY]
    compare_to_baseline_in_loss = option_dict[
        nn_utils.COMPARE_TO_BASELINE_IN_LOSS_KEY
    ]
    num_examples_per_batch = option_dict[nn_utils.BATCH_SIZE_KEY]
    sentinel_value = option_dict[nn_utils.SENTINEL_VALUE_KEY]
    patch_size_2pt5km_pixels = option_dict[nn_utils.PATCH_SIZE_KEY]
    patch_buffer_size_2pt5km_pixels = option_dict[
        nn_utils.PATCH_BUFFER_SIZE_KEY
    ]
    resid_baseline_model_name = option_dict[nn_utils.RESID_BASELINE_MODEL_KEY]
    resid_baseline_model_dir_name = option_dict[
        nn_utils.RESID_BASELINE_MODEL_DIR_KEY
    ]
    resid_baseline_lead_time_hours = option_dict[
        nn_utils.RESID_BASELINE_LEAD_TIME_KEY
    ]

    error_checking.assert_is_integer(patch_overlap_size_2pt5km_pixels)
    error_checking.assert_is_geq(patch_overlap_size_2pt5km_pixels, 16)

    # Create mask for use in loss function.  Recall that, when you do patchwise
    # training, each patch has an inner domain and outer domain -- and model
    # predictions should be evaluated only on the inner domain.  This Boolean
    # mask tells the loss function which pixels are part of the inner domain (1)
    # and which are part of the outer domain (0).
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

    # Read normalization file.
    print('Reading normalization parameters from: "{0:s}"...'.format(
        nwp_normalization_file_name
    ))
    nwp_norm_param_table_xarray = nwp_model_io.read_normalization_file(
        nwp_normalization_file_name
    )

    # Find all forecast-initialization times in training period.
    nwp_model_names = list(nwp_model_to_dir_name.keys())
    nwp_model_names.sort()

    init_times_unix_sec = nn_utils.find_relevant_init_times(
        first_time_by_period_unix_sec=
        numpy.array([first_init_time_unix_sec], dtype=int),
        last_time_by_period_unix_sec=
        numpy.array([last_init_time_unix_sec], dtype=int),
        nwp_model_names=nwp_model_names
    )

    # Do some housekeeping before entering the main generator loop.
    numpy.random.shuffle(init_times_unix_sec)
    init_time_index = 0

    patch_metalocation_dict = init_patch_metalocation_dict(
        patch_size_2pt5km_pixels=patch_size_2pt5km_pixels,
        patch_overlap_size_2pt5km_pixels=patch_overlap_size_2pt5km_pixels
    )

    full_grid_target_matrix = None
    full_grid_baseline_matrix = None
    full_grid_predictor_matrix_2pt5km = None
    num_target_fields = len(target_field_names)

    # Enter the main generator loop, captain.  For every iteration through this
    # loop, the generator yields one batch of data (see the "yield" statement
    # at the end).
    while True:

        # Initialize matrices (numpy arrays) for the current batch.
        dummy_patch_location_dict = misc_utils.determine_patch_locations(
            patch_size_2pt5km_pixels=patch_size_2pt5km_pixels
        )

        matrix_dict = nn_utils.init_matrices_1batch(
            nwp_model_names=nwp_model_names,
            nwp_model_to_field_names=nwp_model_to_field_names,
            num_nwp_lead_times=1,
            target_field_names=target_field_names,
            num_target_lag_times=0,
            num_recent_bias_times=0,
            num_examples_per_batch=num_examples_per_batch,
            patch_location_dict=dummy_patch_location_dict,
            do_residual_prediction=False
        )
        predictor_matrix_2pt5km = matrix_dict[
            nn_utils.PREDICTOR_MATRIX_2PT5KM_KEY
        ]
        target_matrix = matrix_dict[nn_utils.TARGET_MATRIX_KEY]

        if compare_to_baseline_in_loss:
            new_dims = target_matrix.shape[:-1] + (2 * num_target_fields,)
            target_matrix = numpy.full(new_dims, numpy.nan)

        num_examples_in_memory = 0

        # As long as `num_examples_in_memory < num_examples_per_batch`, the
        # current batch needs more data.
        while num_examples_in_memory < num_examples_per_batch:

            # Update the "patch-metalocation dictionary," which moves the patch
            # to a new location within the full NBM grid.  This ensures that the
            # model is trained with patches coming from everywhere in the NBM
            # grid.
            patch_metalocation_dict = update_patch_metalocation_dict(
                patch_metalocation_dict
            )

            # If the start row is negative, this means that, for the current
            # initialization time, the patch has gone to all possible locations
            # in the NBM grid.  Thus, we need to read predictor data (raw NWP
            # forecasts) for a new initialization time.
            if patch_metalocation_dict[PATCH_START_ROW_KEY] < 0:
                full_grid_target_matrix = None
                full_grid_baseline_matrix = None
                full_grid_predictor_matrix_2pt5km = None

                init_time_index, init_times_unix_sec = (
                    nn_utils.increment_init_time(
                        current_index=init_time_index,
                        init_times_unix_sec=init_times_unix_sec
                    )
                )
                continue

            # If `full_grid_target_matrix is None`, we must read data for a new
            # forecast-initialization time.
            try:
                if full_grid_target_matrix is None:
                    full_grid_target_matrix = nn_utils.read_targets_one_example(
                        init_time_unix_sec=init_times_unix_sec[init_time_index],
                        target_lead_time_hours=target_lead_time_hours,
                        target_field_names=target_field_names,
                        target_dir_name=target_dir_name,
                        target_norm_param_table_xarray=None,
                        target_resid_norm_param_table_xarray=None,
                        use_quantile_norm=False,
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

            # If `full_grid_target_matrix is None` still, this means something
            # went wrong while trying to read data (it happens rarely), so let's
            # just continue to the next initialization time in the list.
            if full_grid_target_matrix is None:
                full_grid_target_matrix = None
                full_grid_baseline_matrix = None
                full_grid_predictor_matrix_2pt5km = None

                init_time_index, init_times_unix_sec = (
                    nn_utils.increment_init_time(
                        current_index=init_time_index,
                        init_times_unix_sec=init_times_unix_sec
                    )
                )
                continue

            # If `full_grid_baseline_matrix is None`, we must read data for a
            # new forecast-initialization time.
            try:
                if (
                        compare_to_baseline_in_loss and
                        full_grid_baseline_matrix is None
                ):
                    full_grid_baseline_matrix = (
                        nwp_input.read_residual_baseline_one_example(
                            init_time_unix_sec=
                            init_times_unix_sec[init_time_index],
                            nwp_model_name=resid_baseline_model_name,
                            nwp_lead_time_hours=
                            resid_baseline_lead_time_hours,
                            nwp_directory_name=
                            resid_baseline_model_dir_name,
                            target_field_names=target_field_names,
                            patch_location_dict=None,
                            predict_dewpoint_depression=False,
                            predict_gust_excess=False
                        )
                    )

                    full_grid_target_matrix = numpy.concatenate(
                        [full_grid_target_matrix, full_grid_baseline_matrix],
                        axis=-1
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

            # If `full_grid_baseline_matrix is None` still, this means something
            # went wrong while trying to read data (it happens rarely), so let's
            # just continue to the next initialization time in the list.
            if (
                    compare_to_baseline_in_loss and
                    full_grid_baseline_matrix is None
            ):
                full_grid_target_matrix = None
                full_grid_baseline_matrix = None
                full_grid_predictor_matrix_2pt5km = None

                init_time_index, init_times_unix_sec = (
                    nn_utils.increment_init_time(
                        current_index=init_time_index,
                        init_times_unix_sec=init_times_unix_sec
                    )
                )
                continue

            # If `full_grid_predictor_matrix_2pt5km is None`, we must read data
            # for a new forecast-initialization time.
            try:
                if full_grid_predictor_matrix_2pt5km is None:
                    (
                        full_grid_predictor_matrix_2pt5km,
                        _, _, _,
                        found_any_predictors,
                        _
                    ) = nwp_input.read_predictors_one_example(
                        init_time_unix_sec=init_times_unix_sec[init_time_index],
                        nwp_model_names=nwp_model_names,
                        nwp_lead_times_hours=
                        numpy.array([nwp_lead_time_hours], dtype=int),
                        nwp_model_to_field_names=nwp_model_to_field_names,
                        nwp_model_to_dir_name=nwp_model_to_dir_name,
                        nwp_norm_param_table_xarray=nwp_norm_param_table_xarray,
                        nwp_resid_norm_param_table_xarray=None,
                        use_quantile_norm=True,
                        backup_nwp_model_name=backup_nwp_model_name,
                        backup_nwp_directory_name=backup_nwp_directory_name,
                        patch_location_dict=None
                    )
                else:
                    found_any_predictors = True
            except:
                warning_string = (
                    'POTENTIAL ERROR: Could not read predictors for init time '
                    '{0:s}.  Something went wrong in '
                    '`nwp_input.read_predictors_one_example`.'
                ).format(
                    time_conversion.unix_sec_to_string(
                        init_times_unix_sec[init_time_index], '%Y-%m-%d-%H'
                    )
                )

                warnings.warn(warning_string)
                found_any_predictors = False

            # If no predictors were found, this means something went wrong while
            # trying to read data (it happens rarely), so let's just continue to
            # the next initialization time in the list.
            if not found_any_predictors:
                full_grid_target_matrix = None
                full_grid_baseline_matrix = None
                full_grid_predictor_matrix_2pt5km = None

                init_time_index, init_times_unix_sec = (
                    nn_utils.increment_init_time(
                        current_index=init_time_index,
                        init_times_unix_sec=init_times_unix_sec
                    )
                )
                continue

            # If we have gotten this far through the inner loop, we have data on
            # the full NBM grid!  Now we just need to extract a patch.
            patch_location_dict = misc_utils.determine_patch_locations(
                patch_size_2pt5km_pixels=patch_size_2pt5km_pixels,
                start_row_2pt5km=patch_metalocation_dict[PATCH_START_ROW_KEY],
                start_column_2pt5km=
                patch_metalocation_dict[PATCH_START_COLUMN_KEY]
            )
            pld = patch_location_dict

            j_start = pld[misc_utils.ROW_LIMITS_2PT5KM_KEY][0]
            j_end = pld[misc_utils.ROW_LIMITS_2PT5KM_KEY][1] + 1
            k_start = pld[misc_utils.COLUMN_LIMITS_2PT5KM_KEY][0]
            k_end = pld[misc_utils.COLUMN_LIMITS_2PT5KM_KEY][1] + 1
            i = num_examples_in_memory + 0

            target_matrix[i, ...] = (
                full_grid_target_matrix[j_start:j_end, k_start:k_end, ...]
            )

            # For earlier URMA data (~2017), there are sometimes NaN values
            # along the edge of the full NBM grid.  I think this is because
            # earlier URMA data just had a smaller domain.  In any case, if we
            # got a patch with NaN's, this means some of the patch has no
            # correct answers for training, so we just continue to the next
            # patch location.
            # if numpy.any(numpy.isnan(
            #         target_matrix[i, ..., :num_target_fields]
            # )):
            #     continue

            # TODO(thunderhoser): This can also happen along edges of grid for
            # residual baseline, if residual baseline is high-res ensemble.
            if numpy.any(numpy.isnan(
                    target_matrix[i, num_target_fields:(2 * num_target_fields)]
            )):
                print('Residual baseline contains NaN values ({0:.6f}%).'.format(
                    100 * numpy.mean(numpy.isnan(
                        target_matrix[i, num_target_fields:(2 * num_target_fields)]
                    ))
                ))
            else:
                print('Residual baseline does NOT contain NaN values.')

            if numpy.any(numpy.isnan(target_matrix[i, ...])):
                continue

            predictor_matrix_2pt5km[i, ...] = (
                full_grid_predictor_matrix_2pt5km[j_start:j_end, k_start:k_end, ...]
            )

            num_examples_in_memory += 1

        # Add the Boolean mask to the target matrix (as the last channel).
        target_matrix = numpy.concatenate(
            [target_matrix, mask_matrix_for_loss], axis=-1
        )

        # Report data properties.
        predictor_matrices = nn_utils.create_data_dict_or_tuple(
            predictor_matrix_2pt5km=predictor_matrix_2pt5km,
            nbm_constant_matrix=None,
            predictor_matrix_lagged_targets=None,
            predictor_matrix_10km=None,
            predictor_matrix_20km=None,
            predictor_matrix_40km=None,
            recent_bias_matrix_2pt5km=None,
            recent_bias_matrix_10km=None,
            recent_bias_matrix_20km=None,
            recent_bias_matrix_40km=None,
            predictor_matrix_resid_baseline=None,
            target_matrix=target_matrix,
            sentinel_value=sentinel_value,
            return_predictors_as_dict=False
        )

        # Yield the current batch of data.
        yield predictor_matrices[0], target_matrix


def train_u_net(
        model_object, num_epochs,
        num_training_batches_per_epoch, training_option_dict,
        num_validation_batches_per_epoch, validation_option_dict,
        loss_function_string, optimizer_function_string,
        metric_function_strings, u_net_architecture_dict,
        plateau_patience_epochs, plateau_learning_rate_multiplier,
        early_stopping_patience_epochs, patch_overlap_size_2pt5km_pixels,
        output_dir_name):
    """Trains simple U-net.

    :param model_object: See documentation for `train_model`.
    :param num_epochs: Same.
    :param num_training_batches_per_epoch: Same.
    :param training_option_dict: Same.
    :param num_validation_batches_per_epoch: Same.
    :param validation_option_dict: Same.
    :param loss_function_string: Same.
    :param optimizer_function_string: Same.
    :param metric_function_strings: Same.
    :param u_net_architecture_dict: Same.
    :param plateau_patience_epochs: Same.
    :param plateau_learning_rate_multiplier: Same.
    :param early_stopping_patience_epochs: Same.
    :param patch_overlap_size_2pt5km_pixels: Same.
    :param output_dir_name: Same.
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
        'first_init_time_unix_sec', 'last_init_time_unix_sec'
    ]
    for this_key in list(training_option_dict.keys()):
        if this_key in validation_keys_to_keep:
            continue

        validation_option_dict[this_key] = training_option_dict[this_key]

    training_option_dict = nn_utils.check_u_net_generator_args(
        training_option_dict
    )
    validation_option_dict = nn_utils.check_u_net_generator_args(
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

    training_generator = data_generator_for_u_net(
        option_dict=training_option_dict,
        patch_overlap_size_2pt5km_pixels=patch_overlap_size_2pt5km_pixels
    )
    validation_generator = data_generator_for_u_net(
        option_dict=validation_option_dict,
        patch_overlap_size_2pt5km_pixels=patch_overlap_size_2pt5km_pixels
    )

    metafile_name = nn_utils.find_metafile(
        model_file_name=model_file_name, raise_error_if_missing=False
    )
    print('Writing metadata to: "{0:s}"...'.format(metafile_name))

    nn_utils.write_metafile(
        pickle_file_name=metafile_name,
        num_epochs=num_epochs,
        use_exp_moving_average_with_decay=False,
        num_training_batches_per_epoch=num_training_batches_per_epoch,
        training_option_dict=training_option_dict,
        num_validation_batches_per_epoch=num_validation_batches_per_epoch,
        validation_option_dict=validation_option_dict,
        loss_function_string=loss_function_string,
        optimizer_function_string=optimizer_function_string,
        metric_function_strings=metric_function_strings,
        u_net_architecture_dict=u_net_architecture_dict,
        chiu_net_architecture_dict=None,
        chiu_net_pp_architecture_dict=None,
        chiu_next_pp_architecture_dict=None,
        plateau_patience_epochs=plateau_patience_epochs,
        plateau_learning_rate_multiplier=plateau_learning_rate_multiplier,
        early_stopping_patience_epochs=early_stopping_patience_epochs,
        patch_overlap_fast_gen_2pt5km_pixels=patch_overlap_size_2pt5km_pixels
    )

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


def train_model(
        model_object, num_epochs, use_exp_moving_average_with_decay,
        num_training_batches_per_epoch, training_option_dict,
        num_validation_batches_per_epoch, validation_option_dict,
        loss_function_string, optimizer_function_string,
        metric_function_strings,
        u_net_architecture_dict, chiu_net_architecture_dict,
        chiu_net_pp_architecture_dict, chiu_next_pp_architecture_dict,
        plateau_patience_epochs, plateau_learning_rate_multiplier,
        early_stopping_patience_epochs, patch_overlap_fast_gen_2pt5km_pixels,
        output_dir_name,
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
    :param patch_overlap_fast_gen_2pt5km_pixels: See documentation for
        `data_generator`.
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
        training_generator = data_generator(
            option_dict=training_option_dict,
            patch_overlap_size_2pt5km_pixels=
            patch_overlap_fast_gen_2pt5km_pixels
        )
        validation_generator = data_generator(
            option_dict=validation_option_dict,
            patch_overlap_size_2pt5km_pixels=
            patch_overlap_fast_gen_2pt5km_pixels
        )

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
        patch_overlap_fast_gen_2pt5km_pixels=
        patch_overlap_fast_gen_2pt5km_pixels
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
        model_object, full_predictor_matrices, num_examples_per_batch,
        model_metadata_dict, patch_overlap_size_2pt5km_pixels,
        use_trapezoidal_weighting=False, verbose=True):
    """Applies neural net trained with multi-patch strategy to the full grid.

    E = number of examples
    M = number of rows in full grid
    N = number of columns in full grid
    F = number of target fields
    S = ensemble size

    :param model_object: Trained neural net (instance of `keras.models.Model`).
    :param full_predictor_matrices: See output doc for `data_generator` --
        except these matrices are on the full grid, not one patch.
    :param num_examples_per_batch: Batch size.
    :param model_metadata_dict: Dictionary returned by
        `neural_net_utils.read_metafile`.
    :param patch_overlap_size_2pt5km_pixels: Overlap between adjacent patches,
        measured in number of pixels on the finest-resolution (2.5-km) grid.
    :param use_trapezoidal_weighting: Boolean flag.  If True, trapezoidal
        weighting will be used, so that predictions in the center of a given
        patch are given a higher weight than predictions at the edge.
    :param verbose: Boolean flag.  If True, will print progress messages.
    :return: prediction_matrix: E-by-M-by-N-by-F-by-S numpy array of predicted
        values.
    """

    # Check input args.
    these_dim = model_object.layers[-1].output.shape
    num_rows_in_patch = these_dim[1]
    num_columns_in_patch = these_dim[2]
    num_target_fields = these_dim[3]

    # TODO(thunderhoser): Might relax this constraint eventually -- I don't
    # know.
    error_checking.assert_equals(num_rows_in_patch, num_columns_in_patch)
    error_checking.assert_is_boolean(use_trapezoidal_weighting)

    error_checking.assert_is_integer(patch_overlap_size_2pt5km_pixels)
    error_checking.assert_is_geq(patch_overlap_size_2pt5km_pixels, 0)
    error_checking.assert_is_less_than(
        patch_overlap_size_2pt5km_pixels,
        min([num_rows_in_patch, num_columns_in_patch])
    )

    if use_trapezoidal_weighting:
        half_num_rows_in_patch = int(numpy.ceil(
            float(num_rows_in_patch) / 2
        ))
        half_num_columns_in_patch = int(numpy.ceil(
            float(num_columns_in_patch) / 2
        ))
        error_checking.assert_is_geq(
            patch_overlap_size_2pt5km_pixels,
            max([half_num_rows_in_patch, half_num_columns_in_patch])
        )

    error_checking.assert_is_boolean(verbose)

    # Do actual stuff.
    num_rows_2pt5km = len(nbm_utils.NBM_Y_COORDS_METRES)
    num_columns_2pt5km = len(nbm_utils.NBM_X_COORDS_METRES)

    patch_metalocation_dict = init_patch_metalocation_dict(
        patch_size_2pt5km_pixels=num_rows_in_patch,
        patch_overlap_size_2pt5km_pixels=patch_overlap_size_2pt5km_pixels
    )

    validation_option_dict = model_metadata_dict[
        nn_utils.VALIDATION_OPTIONS_KEY
    ]
    patch_buffer_size = validation_option_dict[nn_utils.PATCH_BUFFER_SIZE_KEY]

    if use_trapezoidal_weighting:
        weight_matrix = make_trapezoidal_weight_matrix(
            patch_size_2pt5km_pixels=num_rows_in_patch - 2 * patch_buffer_size,
            patch_overlap_size_2pt5km_pixels=patch_overlap_size_2pt5km_pixels
        )
        weight_matrix = numpy.pad(
            weight_matrix, pad_width=patch_buffer_size,
            mode='constant', constant_values=0
        )
    else:
        weight_matrix = nn_utils.patch_buffer_to_mask(
            patch_size_2pt5km_pixels=num_rows_in_patch,
            patch_buffer_size_2pt5km_pixels=patch_buffer_size
        )

    weight_matrix = numpy.expand_dims(weight_matrix, axis=0)
    weight_matrix = numpy.expand_dims(weight_matrix, axis=-1)
    weight_matrix = numpy.expand_dims(weight_matrix, axis=-1)

    num_examples = full_predictor_matrices[0].shape[0]
    these_dim = (
        num_examples, num_rows_2pt5km, num_columns_2pt5km, num_target_fields, 1
    )
    prediction_count_matrix = numpy.full(these_dim, 0, dtype=float)
    summed_prediction_matrix = None
    # summed_prediction_matrix = numpy.full(these_dim, 0.)

    while True:
        patch_metalocation_dict = update_patch_metalocation_dict(
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

        patch_prediction_matrix = nn_training_simple.apply_model(
            model_object=model_object,
            predictor_matrices=patch_predictor_matrices,
            num_examples_per_batch=num_examples_per_batch,
            target_field_names=
            validation_option_dict[nn_utils.TARGET_FIELDS_KEY],
            verbose=False
        )

        if summed_prediction_matrix is None:
            ensemble_size = patch_prediction_matrix.shape[-1]
            these_dim = (
                num_examples, num_rows_2pt5km, num_columns_2pt5km,
                num_target_fields, ensemble_size
            )
            summed_prediction_matrix = numpy.full(these_dim, 0.)

        summed_prediction_matrix[:, i_start:i_end, j_start:j_end, ...] += (
                weight_matrix * patch_prediction_matrix
        )
        prediction_count_matrix[:, i_start:i_end, j_start:j_end, ...] += (
            weight_matrix
        )

    if verbose:
        print('Have applied model everywhere in full grid!')

    prediction_count_matrix = prediction_count_matrix.astype(float)
    prediction_count_matrix[prediction_count_matrix < TOLERANCE] = numpy.nan
    return summed_prediction_matrix / prediction_count_matrix

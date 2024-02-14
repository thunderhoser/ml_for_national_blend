"""Helper methods for training a neural network."""

import os
import sys
import warnings
import numpy

THIS_DIRECTORY_NAME = os.path.dirname(os.path.realpath(
    os.path.join(os.getcwd(), os.path.expanduser(__file__))
))
sys.path.append(os.path.normpath(os.path.join(THIS_DIRECTORY_NAME, '..')))

import time_conversion
import time_periods
import error_checking
import interp_nwp_model_io
import urma_io
import nwp_model_utils
import urma_utils

TIME_FORMAT = '%Y-%m-%d-%H'
HOURS_TO_SECONDS = 3600

INIT_TIME_LIMITS_KEY = 'init_time_limits_unix_sec'
NWP_LEAD_TIMES_KEY = 'nwp_lead_times_hours'
NWP_MODEL_TO_DIR_KEY = 'nwp_model_to_dir_name'
NWP_MODEL_TO_FIELDS_KEY = 'nwp_model_to_field_names'
TARGET_LEAD_TIME_KEY = 'target_lead_time_hours'
TARGET_FIELDS_KEY = 'target_field_names'
TARGET_DIR_KEY = 'target_dir_name'
BATCH_SIZE_KEY = 'num_examples_per_batch'
SENTINEL_VALUE_KEY = 'sentinel_value'

DEFAULT_GENERATOR_OPTION_DICT = {
    SENTINEL_VALUE_KEY: -10.
}


def _check_generator_args(option_dict):
    """Checks input arguments for generator.

    :param option_dict: See doc for `data_generator`.
    :return: option_dict: Same as input, except defaults may have been added.
    """

    orig_option_dict = option_dict.copy()
    option_dict = DEFAULT_GENERATOR_OPTION_DICT.copy()
    option_dict.update(orig_option_dict)

    error_checking.assert_is_numpy_array(
        option_dict[INIT_TIME_LIMITS_KEY],
        exact_dimensions=numpy.array([2], dtype=int)
    )
    error_checking.assert_is_integer_numpy_array(
        option_dict[INIT_TIME_LIMITS_KEY]
    )
    error_checking.assert_is_greater_numpy_array(
        numpy.diff(option_dict[INIT_TIME_LIMITS_KEY]),
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

    error_checking.assert_is_integer(option_dict[TARGET_LEAD_TIME_KEY])
    error_checking.assert_is_greater(option_dict[TARGET_LEAD_TIME_KEY], 0)

    error_checking.assert_is_string_list(option_dict[TARGET_FIELDS_KEY])
    for this_field_name in option_dict[TARGET_FIELDS_KEY]:
        urma_utils.check_field_name(this_field_name)

    error_checking.assert_is_string(option_dict[TARGET_DIR_KEY])

    error_checking.assert_is_integer(option_dict[BATCH_SIZE_KEY])
    # error_checking.assert_is_geq(option_dict[BATCH_SIZE_KEY], 8)

    error_checking.assert_is_not_nan(option_dict[SENTINEL_VALUE_KEY])

    return option_dict


def _init_predictor_matrices_1example(
        nwp_model_names, nwp_model_to_field_names, num_nwp_lead_times):
    """Initializes predictor matrices for one example.

    :param nwp_model_names: 1-D list with names of NWP models.
    :param nwp_model_to_field_names: Dictionary.  For details, see documentation
        for `data_generator`.
    :param num_nwp_lead_times: Number of lead times.
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
        num_target_fields, num_examples_per_batch):
    """Initializes predictor and target matrices for one batch.

    :param nwp_model_names: 1-D list with names of NWP models.
    :param nwp_model_to_field_names: Dictionary.  For details, see documentation
        for `data_generator`.
    :param num_nwp_lead_times: Number of lead times.
    :param num_target_fields: Number of target fields.
    :param num_examples_per_batch: Batch size.
    :return: predictor_matrix_2pt5km: numpy array for NWP data with 2.5-km
        resolution.  If there are no 2.5-km models, this is None instead of an
        array.
    :return: predictor_matrix_10km: Same but for 10-km models.
    :return: predictor_matrix_20km: Same but for 20-km models.
    :return: predictor_matrix_40km: Same but for 40-km models.
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

    target_matrix = numpy.full(
        (num_examples_per_batch, num_rows, num_columns, num_target_fields),
        numpy.nan
    )

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
        target_matrix
    )


def _read_targets_one_example(
        init_time_unix_sec, target_lead_time_hours, target_field_names,
        target_dir_name):
    """Reads target fields for one example.

    NBM = National Blend of Models

    M = number of rows in NBM grid (2.5-km target grid)
    N = number of columns in NBM grid (2.5-km target grid)
    F = number of target fields

    :param init_time_unix_sec: Forecast-initialization time.
    :param target_lead_time_hours: See documentation for `data_generator`.
    :param target_field_names: Same.
    :param target_dir_name: Same.
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

    return numpy.transpose(
        urma_table_xarray[urma_utils.DATA_KEY].values[0, ...],
        axes=(1, 0, 2)
    )


def _read_predictors_one_example(
        init_time_unix_sec, nwp_model_names, nwp_lead_times_hours,
        nwp_model_to_field_names, nwp_model_to_dir_name):
    """Reads predictor fields for one example.

    :param init_time_unix_sec: Forecast-initialization time.
    :param nwp_model_names: 1-D list with names of NWP models used to create
        predictors.
    :param nwp_lead_times_hours: See documentation for `data_generator`.
    :param nwp_model_to_field_names: Same.
    :param nwp_model_to_dir_name: Same.
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
        num_nwp_lead_times=num_nwp_lead_times
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
            print(nwp_forecast_table_xarray.coords['field_name'].values)
            print(nwp_model_to_field_names[nwp_model_names[i]])
            nwp_forecast_table_xarray = nwp_model_utils.subset_by_field(
                nwp_forecast_table_xarray=nwp_forecast_table_xarray,
                desired_field_names=nwp_model_to_field_names[nwp_model_names[i]]
            )

            matrix_index = numpy.sum(
                nwp_downsampling_factors[:i] == nwp_downsampling_factors[i]
            )

            if nwp_downsampling_factors[i] == 1:
                predictor_matrices_2pt5km[matrix_index][..., j, :] = (
                    nwp_forecast_table_xarray[nwp_model_utils.DATA_KEY].values[0, ...]
                )
            elif nwp_downsampling_factors[i] == 4:
                predictor_matrices_10km[matrix_index][..., j, :] = (
                    nwp_forecast_table_xarray[nwp_model_utils.DATA_KEY].values[0, ...]
                )
            elif nwp_downsampling_factors[i] == 8:
                predictor_matrices_20km[matrix_index][..., j, :] = (
                    nwp_forecast_table_xarray[nwp_model_utils.DATA_KEY].values[0, ...]
                )
            else:
                predictor_matrices_40km[matrix_index][..., j, :] = (
                    nwp_forecast_table_xarray[nwp_model_utils.DATA_KEY].values[0, ...]
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


def data_generator(option_dict):
    """Generates training or validation data for neural network.

    Generators should be used only at training time, not at inference time.

    NBM = National Blend of Models

    E = number of examples per batch = "batch size"
    M = number of rows in NBM grid (2.5-km target grid)
    N = number of columns in NBM grid (2.5-km target grid)
    P = number of NWP fields (predictor variables) at 2.5-km resolution
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
    option_dict["init_time_limits_unix_sec"]: length-2 numpy array with first
        and last init times to be used.
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
    option_dict["target_lead_time_hours"]: Lead time for target fields.
    option_dict["target_field_names"]: length-F list with names of target
        fields.  Each must be accepted by `urma_utils.check_field_name`.
    option_dict["target_dir_name"]: Path to directory with target fields (i.e.,
        URMA data).  Files within this directory will be found by
        `urma_io.find_file` and read by `urma_io.read_file`.
    option_dict["num_examples_per_batch"]: Number of data examples per batch,
        usually just called "batch size".
    option_dict["sentinel_value"]: All NaN will be replaced with this value.

    :return: predictor_matrices: List with the following items.  Some items may
        be missing.

    predictor_matrices[0]: E-by-M-by-N-by-L-by-P numpy array of predictors at
        2.5-km resolution.
    predictor_matrices[1]: E-by-m-by-n-by-L-by-p numpy array of predictors at
        10-km resolution.
    predictor_matrices[2]: E-by-mm-by-nn-by-L-by-pp numpy array of predictors at
        20-km resolution.
    predictor_matrices[3]: E-by-mmm-by-nnn-by-L-by-ppp numpy array of predictors
        at 40-km resolution.

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

    # TODO(thunderhoser): Still need to do normalization!

    # TODO(thunderhoser): Still need to add constant fields!

    # TODO(thunderhoser): According to the project proposal, target fields are
    # supposed to include "minimum RH" and "maximum RH".  But there are two
    # problems: (1) I don't know what we're minning/maxxing over -- is it
    # min/max over the hour at every grid point?  (2) This variable is not
    # available in URMA, so we have no ground truth.

    option_dict = _check_generator_args(option_dict)
    init_time_limits_unix_sec = option_dict[INIT_TIME_LIMITS_KEY]
    nwp_lead_times_hours = option_dict[NWP_LEAD_TIMES_KEY]
    nwp_model_to_dir_name = option_dict[NWP_MODEL_TO_DIR_KEY]
    nwp_model_to_field_names = option_dict[NWP_MODEL_TO_FIELDS_KEY]
    target_lead_time_hours = option_dict[TARGET_LEAD_TIME_KEY]
    target_field_names = option_dict[TARGET_FIELDS_KEY]
    target_dir_name = option_dict[TARGET_DIR_KEY]
    num_examples_per_batch = option_dict[BATCH_SIZE_KEY]
    sentinel_value = option_dict[SENTINEL_VALUE_KEY]

    first_nwp_model_names = list(nwp_model_to_dir_name.keys())
    second_nwp_model_names = list(nwp_model_to_field_names.keys())
    assert set(first_nwp_model_names) == set(second_nwp_model_names)

    nwp_model_names = list(set(first_nwp_model_names))
    nwp_model_names = [
        m for m in nwp_model_names if m != nwp_model_utils.WRF_ARW_MODEL_NAME
    ]

    # TODO(thunderhoser): Different NWP models are available at different init
    # times.  Currently, I handle this by using only common init times.
    # However, I should eventually come up with something more clever.
    init_time_intervals_sec = numpy.array([
        nwp_model_utils.model_to_init_time_interval(m) for m in nwp_model_names
    ], dtype=int)

    init_times_unix_sec = time_periods.range_and_interval_to_list(
        start_time_unix_sec=init_time_limits_unix_sec[0],
        end_time_unix_sec=init_time_limits_unix_sec[-1],
        time_interval_sec=numpy.max(init_time_intervals_sec),
        include_endpoint=True
    )

    # Do actual stuff.
    init_time_index = len(init_times_unix_sec)

    while True:
        (
            predictor_matrix_2pt5km, predictor_matrix_10km,
            predictor_matrix_20km, predictor_matrix_40km,
            target_matrix
        ) = _init_matrices_1batch(
            nwp_model_names=nwp_model_names,
            nwp_model_to_field_names=nwp_model_to_field_names,
            num_nwp_lead_times=len(nwp_lead_times_hours),
            num_target_fields=len(target_field_names),
            num_examples_per_batch=num_examples_per_batch
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
                target_dir_name=target_dir_name
            )

            if this_target_matrix is None:
                init_time_index += 1
                continue

            i = num_examples_in_memory + 0
            target_matrix[i, ...] = this_target_matrix

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
                nwp_model_to_dir_name=nwp_model_to_dir_name
            )

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

        predictor_matrices = [
            m for m in [
                predictor_matrix_2pt5km, predictor_matrix_10km,
                predictor_matrix_20km, predictor_matrix_40km
            ]
            if m is not None
        ]

        predictor_matrices = [p.astype('float32') for p in predictor_matrices]
        yield predictor_matrices, target_matrix

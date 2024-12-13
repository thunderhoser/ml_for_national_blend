"""Methods for computing spread-skill relationship."""

import os
import sys
import copy
import numpy
import xarray

THIS_DIRECTORY_NAME = os.path.dirname(os.path.realpath(
    os.path.join(os.getcwd(), os.path.expanduser(__file__))
))
sys.path.append(os.path.normpath(os.path.join(THIS_DIRECTORY_NAME, '..')))

import file_system_utils
import error_checking
import prediction_io

TOLERANCE = 1e-6

BIN_DIM = 'spread_bin'
BIN_EDGE_DIM = 'spread_bin_edge'
LATITUDE_DIM = 'grid_row'
LONGITUDE_DIM = 'grid_column'
FIELD_DIM = 'field'

SSREL_KEY = 'spread_skill_reliability'
SSRAT_KEY = 'spread_skill_ratio'
MEAN_PREDICTION_STDEV_KEY = 'mean_prediction_stdev'
RMSE_KEY = 'root_mean_squared_error'
EXAMPLE_COUNT_KEY = 'example_count'
MEAN_DETERMINISTIC_PRED_KEY = 'mean_deterministic_prediction'
MEAN_TARGET_KEY = 'mean_target_value'
BIN_EDGE_PREDICTION_STDEV_KEY = 'bin_edge_prediction_stdev'

PREDICTION_FILES_KEY = 'prediction_file_names'


def memory_efficient_stdev(input_matrix):
    """Memory-efficient calculation of standard deviation (along final axis).

    S = ensemble size

    :param input_matrix: numpy array, where the last axis has length S.
    :return: stdev_matrix: numpy array of standard deviations, with the same
        size as `input_array`, except the last axis is gone.
    """

    ensemble_size = input_matrix.shape[-1]
    error_checking.assert_is_greater(ensemble_size, 1)

    mean_matrix = numpy.nanmean(input_matrix, axis=-1)

    variance_matrix = (input_matrix[..., 0] - mean_matrix) ** 2
    ensemble_size_matrix = numpy.invert(
        numpy.isnan(input_matrix[..., 0])
    ).astype(float)

    for k in range(1, ensemble_size):
        variance_matrix += (
            numpy.nan_to_num(input_matrix[..., k], nan=0.) - mean_matrix
        ) ** 2

        ensemble_size_matrix += numpy.invert(
            numpy.isnan(input_matrix[..., k])
        ).astype(float)

    ensemble_size_matrix[ensemble_size_matrix < 2] = numpy.nan
    variance_matrix /= (ensemble_size_matrix - 1)
    return numpy.sqrt(variance_matrix)


def read_inputs(prediction_file_names, target_field_names):
    """Reads inputs (predictions and targets) from many files.

    E = number of examples
    M = number of rows in grid
    N = number of columns in grid
    S = ensemble size

    :param prediction_file_names: 1-D list of paths to prediction files.
        Each file will be read by `prediction_io.read_file`.
    :param target_field_names: 1-D list of field names desired.
    :return: prediction_matrix: E-by-M-by-N-by-1-by-S numpy array of
        predictions.
    :return: target_matrix: E-by-M-by-N-by-1 numpy array of actual values.
    """

    # TODO(thunderhoser): Put this in prediction_io.py.

    error_checking.assert_is_string_list(prediction_file_names)
    error_checking.assert_is_string_list(target_field_names)

    num_times = len(prediction_file_names)
    prediction_matrix = numpy.array([], dtype=float)
    target_matrix = numpy.array([], dtype=float)
    model_file_name = None

    for i in range(num_times):
        print('Reading data from: "{0:s}"...'.format(prediction_file_names[i]))
        this_prediction_table_xarray = prediction_io.read_file(
            prediction_file_names[i]
        )
        tpt = this_prediction_table_xarray

        if model_file_name is None:
            model_file_name = copy.deepcopy(
                tpt.attrs[prediction_io.MODEL_FILE_KEY]
            )
            num_grid_rows = tpt[prediction_io.PREDICTION_KEY].values.shape[0]
            num_grid_columns = tpt[prediction_io.PREDICTION_KEY].values.shape[1]
            ensemble_size = tpt[prediction_io.PREDICTION_KEY].values.shape[-1]

            these_dim = (
                num_times, num_grid_rows, num_grid_columns, 1, ensemble_size
            )
            prediction_matrix = numpy.full(
                these_dim, numpy.nan, dtype=numpy.float32
            )
            target_matrix = numpy.full(
                these_dim[:-1], numpy.nan, dtype=numpy.float32
            )

        assert model_file_name == tpt.attrs[prediction_io.MODEL_FILE_KEY]

        these_indices = numpy.array([
            numpy.where(tpt[prediction_io.FIELD_NAME_KEY].values == f)[0][0]
            for f in target_field_names
        ], dtype=int)

        tpt = tpt.isel({prediction_io.FIELD_DIM: these_indices})
        prediction_matrix[i, ...] = tpt[prediction_io.PREDICTION_KEY].values
        target_matrix[i, ...] = tpt[prediction_io.TARGET_KEY].values

    return prediction_matrix, target_matrix


def get_spread_vs_skill(
        prediction_file_names, target_field_names,
        num_bins_by_target, min_bin_edge_by_target, max_bin_edge_by_target,
        min_bin_edge_prctile_by_target, max_bin_edge_prctile_by_target):
    """Computes spread-skill relationship for multiple target fields.

    T = number of target fields

    :param prediction_file_names: 1-D list of paths to prediction files.  Each
        file will be read by `prediction_io.read_file`.
    :param target_field_names: length-T list of field names.
    :param num_bins_by_target: length-T numpy array with number of spread bins
        for each target.
    :param min_bin_edge_by_target: length-T numpy array with minimum spread
        values in spread-skill plot.  If you instead want minimum values to be
        percentiles over the data, make this argument None and use
        `min_bin_edge_prctile_by_target`.
    :param max_bin_edge_by_target: Same as above but for max.
    :param min_bin_edge_prctile_by_target: length-T numpy array with percentile
        level used to determine minimum spread value in plot for each target.
        If you instead want to specify raw values, make this argument None and
        use `min_bin_edge_by_target`.
    :param max_bin_edge_prctile_by_target: Same as above but for max.
    :return: result_table_xarray: xarray table with results (variable and
        dimension names should make the table self-explanatory).
    """

    # Check input args.
    error_checking.assert_is_string_list(prediction_file_names)
    error_checking.assert_is_string_list(target_field_names)

    num_target_fields = len(target_field_names)
    expected_dim = numpy.array([num_target_fields], dtype=int)

    error_checking.assert_is_numpy_array(
        num_bins_by_target, exact_dimensions=expected_dim
    )
    error_checking.assert_is_integer_numpy_array(num_bins_by_target)
    error_checking.assert_is_geq_numpy_array(num_bins_by_target, 10)
    error_checking.assert_is_leq_numpy_array(num_bins_by_target, 1000)

    if (
            min_bin_edge_by_target is None or
            max_bin_edge_by_target is None
    ):
        error_checking.assert_is_numpy_array(
            min_bin_edge_prctile_by_target, exact_dimensions=expected_dim
        )
        error_checking.assert_is_geq_numpy_array(
            min_bin_edge_prctile_by_target, 0.
        )
        error_checking.assert_is_leq_numpy_array(
            min_bin_edge_prctile_by_target, 10.
        )

        error_checking.assert_is_numpy_array(
            max_bin_edge_prctile_by_target, exact_dimensions=expected_dim
        )
        error_checking.assert_is_geq_numpy_array(
            max_bin_edge_prctile_by_target, 90.
        )
        error_checking.assert_is_leq_numpy_array(
            max_bin_edge_prctile_by_target, 100.
        )
    else:
        error_checking.assert_is_numpy_array(
            min_bin_edge_by_target, exact_dimensions=expected_dim
        )
        error_checking.assert_is_numpy_array(
            max_bin_edge_by_target, exact_dimensions=expected_dim
        )

        for j in range(num_target_fields):
            error_checking.assert_is_greater(
                max_bin_edge_by_target[j],
                min_bin_edge_by_target[j]
            )

    # Read the data.
    prediction_matrix, target_matrix = read_inputs(
        prediction_file_names=prediction_file_names,
        target_field_names=target_field_names
    )

    # Set up the output table.
    orig_dimensions = (num_target_fields,)
    orig_dim_keys = (FIELD_DIM,)
    main_data_dict = {
        SSREL_KEY: (
            orig_dim_keys, numpy.full(orig_dimensions, numpy.nan)
        ),
        SSRAT_KEY: (
            orig_dim_keys, numpy.full(orig_dimensions, numpy.nan)
        )
    }

    these_dimensions = orig_dimensions + (numpy.max(num_bins_by_target),)
    these_dim_keys = orig_dim_keys + (BIN_DIM,)
    main_data_dict.update({
        MEAN_PREDICTION_STDEV_KEY: (
            these_dim_keys, numpy.full(these_dimensions, numpy.nan)
        ),
        RMSE_KEY: (
            these_dim_keys, numpy.full(these_dimensions, numpy.nan)
        ),
        EXAMPLE_COUNT_KEY: (
            these_dim_keys, numpy.full(these_dimensions, numpy.nan)
        ),
        MEAN_DETERMINISTIC_PRED_KEY: (
            these_dim_keys, numpy.full(these_dimensions, numpy.nan)
        ),
        MEAN_TARGET_KEY: (
            these_dim_keys, numpy.full(these_dimensions, numpy.nan)
        )
    })

    these_dimensions = orig_dimensions + (numpy.max(num_bins_by_target) + 1,)
    these_dim_keys = orig_dim_keys + (BIN_EDGE_DIM,)

    main_data_dict.update({
        BIN_EDGE_PREDICTION_STDEV_KEY: (
            these_dim_keys, numpy.full(these_dimensions, numpy.nan)
        )
    })

    bin_indices = numpy.linspace(
        0, numpy.max(num_bins_by_target) - 1,
        num=numpy.max(num_bins_by_target), dtype=int
    )
    bin_edge_indices = numpy.linspace(
        0, numpy.max(num_bins_by_target),
        num=numpy.max(num_bins_by_target) + 1, dtype=int
    )
    metadata_dict = {
        FIELD_DIM: target_field_names,
        BIN_DIM: bin_indices,
        BIN_EDGE_DIM: bin_edge_indices
    }

    result_table_xarray = xarray.Dataset(
        data_vars=main_data_dict, coords=metadata_dict
    )
    result_table_xarray.attrs[PREDICTION_FILES_KEY] = ' '.join([
        '{0:s}'.format(f) for f in prediction_file_names
    ])

    # Do actual stuff.
    deterministic_pred_matrix = numpy.nanmean(prediction_matrix, axis=-1)
    prediction_stdev_matrix = memory_efficient_stdev(prediction_matrix)
    squared_error_matrix = (deterministic_pred_matrix - target_matrix) ** 2

    rtx = result_table_xarray

    for k in range(num_target_fields):
        if min_bin_edge_by_target is not None:
            this_min_edge = min_bin_edge_by_target[k] + 0.
            this_max_edge = max_bin_edge_by_target[k] + 0.
        else:
            this_min_edge = numpy.nanpercentile(
                prediction_stdev_matrix[..., k],
                min_bin_edge_prctile_by_target[k]
            )
            this_max_edge = numpy.nanpercentile(
                prediction_stdev_matrix[..., k],
                max_bin_edge_prctile_by_target[k]
            )

        rtx[BIN_EDGE_PREDICTION_STDEV_KEY].values[k, :] = numpy.linspace(
            this_min_edge, this_max_edge,
            num=num_bins_by_target[k] + 1, dtype=float
        )

        for m in range(num_bins_by_target[k]):
            this_flag_matrix = numpy.logical_and(
                prediction_stdev_matrix[..., k] >=
                rtx[BIN_EDGE_PREDICTION_STDEV_KEY].values[k, m],
                prediction_stdev_matrix[..., k] <
                rtx[BIN_EDGE_PREDICTION_STDEV_KEY].values[k, m + 1]
            )

            this_flag_matrix = numpy.logical_and(
                this_flag_matrix,
                numpy.invert(numpy.isnan(squared_error_matrix[..., k]))
            )

            if numpy.sum(this_flag_matrix) == 0:
                rtx[EXAMPLE_COUNT_KEY].values[k, m] = 0.
                continue

            rtx[MEAN_PREDICTION_STDEV_KEY].values[k, m] = numpy.sqrt(numpy.mean(
                prediction_stdev_matrix[..., k][this_flag_matrix] ** 2
            ))
            rtx[RMSE_KEY].values[k, m] = numpy.sqrt(numpy.mean(
                squared_error_matrix[..., k][this_flag_matrix]
            ))
            rtx[EXAMPLE_COUNT_KEY].values[k, m] = numpy.sum(this_flag_matrix)
            rtx[MEAN_DETERMINISTIC_PRED_KEY].values[k, m] = numpy.mean(
                deterministic_pred_matrix[..., k][this_flag_matrix]
            )
            rtx[MEAN_TARGET_KEY].values[k, m] = numpy.mean(
                target_matrix[..., k][this_flag_matrix]
            )

        these_diffs = numpy.absolute(
            rtx[MEAN_PREDICTION_STDEV_KEY].values[k, :] -
            rtx[RMSE_KEY].values[k, :]
        )
        these_diffs[numpy.isnan(these_diffs)] = 0.
        rtx[SSREL_KEY].values[k] = numpy.average(
            these_diffs, weights=rtx[EXAMPLE_COUNT_KEY].values[k, :]
        )

        this_numerator = numpy.sqrt(numpy.nanmean(
            prediction_stdev_matrix[..., k] ** 2
        ))
        this_denominator = numpy.sqrt(numpy.nanmean(
            squared_error_matrix[..., k]
        ))
        rtx[SSRAT_KEY].values[k] = this_numerator / this_denominator

    result_table_xarray = rtx
    return result_table_xarray


def write_results(result_table_xarray, netcdf_file_name):
    """Writes spread-vs.-skill results to NetCDF file.

    :param result_table_xarray: xarray table in format returned by
        `get_results_all_vars`.
    :param netcdf_file_name: Path to output file.
    """

    file_system_utils.mkdir_recursive_if_necessary(file_name=netcdf_file_name)
    result_table_xarray.to_netcdf(
        path=netcdf_file_name, mode='w', format='NETCDF3_64BIT'
    )


def read_results(netcdf_file_name):
    """Reads spread-vs.-skill results from NetCDF file.

    :param netcdf_file_name: Path to input file.
    :return: result_table_xarray: xarray table.  Documentation in the
        xarray table should make values self-explanatory.
    """

    return xarray.open_dataset(netcdf_file_name)

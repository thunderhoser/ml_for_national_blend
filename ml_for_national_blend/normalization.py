"""Helper methods for normalization."""

import os
import sys
import numpy
import xarray
import scipy.stats
from scipy.interpolate import interp1d

THIS_DIRECTORY_NAME = os.path.dirname(os.path.realpath(
    os.path.join(os.getcwd(), os.path.expanduser(__file__))
))
sys.path.append(os.path.normpath(os.path.join(THIS_DIRECTORY_NAME, '..')))

import error_checking
import urma_io
import nbm_constant_io
import interp_nwp_model_io
import urma_utils
import nbm_constant_utils
import nwp_model_utils

TOLERANCE = 1e-6

MIN_CUMULATIVE_DENSITY = 1e-6
MAX_CUMULATIVE_DENSITY = 1. - 1e-6
# MAX_CUMULATIVE_DENSITY = 0.9995  # To account for 16-bit floats.

NUM_VALUES_KEY = 'num_values'
MEAN_VALUE_KEY = 'mean_value'
MEAN_OF_SQUARES_KEY = 'mean_of_squares'
SAMPLE_VALUES_KEY = 'sample_values'

ACCUM_PRECIP_FIELD_NAMES = [nwp_model_utils.PRECIP_NAME]


def _update_norm_params_1var_1file(norm_param_dict, new_data_matrix,
                                   num_sample_values_per_file, file_index):
    """Updates normalization params for one variable, based on one file.

    :param norm_param_dict: Dictionary with the following keys.
    norm_param_dict['num_values']: Number of values on which current
        estimates are based.
    norm_param_dict['mean_value']: Current mean.
    norm_param_dict['mean_of_squares']: Current mean of squared values.
    norm_param_dict['sample_values']: 1-D numpy array of sample values.

    :param new_data_matrix: numpy array with new values.  Will be used to
        update estimates in `norm_param_dict`.
    :param num_sample_values_per_file: Number of sample values to read.
    :param file_index: Index of current file.  If file_index == k, this means
        that the current file is the [k]th in the list.
    :return: norm_param_dict: Same as input, but with new estimates.
    """

    if numpy.all(numpy.isnan(new_data_matrix)):
        return norm_param_dict

    new_num_values = numpy.sum(numpy.invert(numpy.isnan(new_data_matrix)))

    these_means = numpy.array([
        norm_param_dict[MEAN_VALUE_KEY], numpy.nanmean(new_data_matrix)
    ])
    these_weights = numpy.array([
        norm_param_dict[NUM_VALUES_KEY], new_num_values
    ])
    norm_param_dict[MEAN_VALUE_KEY] = numpy.average(
        these_means, weights=these_weights
    )

    these_means = numpy.array([
        norm_param_dict[MEAN_OF_SQUARES_KEY],
        numpy.nanmean(new_data_matrix ** 2)
    ])
    these_weights = numpy.array([
        norm_param_dict[NUM_VALUES_KEY], new_num_values
    ])
    norm_param_dict[MEAN_OF_SQUARES_KEY] = numpy.average(
        these_means, weights=these_weights
    )

    norm_param_dict[NUM_VALUES_KEY] += new_num_values

    first_index = file_index * num_sample_values_per_file
    last_index = first_index + num_sample_values_per_file

    new_values_real = new_data_matrix[
        numpy.invert(numpy.isnan(new_data_matrix))
    ]
    numpy.random.shuffle(new_values_real)

    norm_param_dict[SAMPLE_VALUES_KEY][first_index:last_index] = (
        new_values_real[:num_sample_values_per_file]
    )

    return norm_param_dict


def _get_standard_deviation_1var(norm_param_dict):
    """Computes standard deviation for one variable.

    :param norm_param_dict: See doc for `_update_norm_params_1var_1file`.
    :return: standard_deviation: Standard deviation.
    """

    if norm_param_dict[NUM_VALUES_KEY] == 0:
        return numpy.nan

    multiplier = float(
        norm_param_dict[NUM_VALUES_KEY]
    ) / (norm_param_dict[NUM_VALUES_KEY] - 1)

    return numpy.sqrt(multiplier * (
        norm_param_dict[MEAN_OF_SQUARES_KEY] -
        norm_param_dict[MEAN_VALUE_KEY] ** 2
    ))


def _z_normalize_1var(data_values, reference_mean, reference_stdev):
    """Does z-score normalization for one variable.

    :param data_values: numpy array of data in physical units.
    :param reference_mean: Mean value from reference dataset.
    :param reference_stdev: Standard deviation from reference dataset.
    :return: data_values: Same as input but in z-scores now.
    """

    # TODO(thunderhoser): Still need unit test.

    if numpy.isnan(reference_stdev):
        data_values[:] = 0.
    else:
        data_values = (data_values - reference_mean) / reference_stdev

    return data_values


def _z_denormalize_1var(data_values, reference_mean, reference_stdev):
    """Does z-score *de*normalization for one variable.

    :param data_values: numpy array of data in z-score units.
    :param reference_mean: Mean value from reference dataset.
    :param reference_stdev: Standard deviation from reference dataset.
    :return: data_values: Same as input but in physical units now.
    """

    # TODO(thunderhoser): Still need unit test.

    if numpy.isnan(reference_stdev):
        data_values[:] = reference_mean
    else:
        data_values = reference_mean + reference_stdev * data_values

    return data_values


def _quantile_normalize_1var(data_values, reference_values_1d):
    """Does quantile normalization for one variable.

    :param data_values: numpy array of data in physical units.
    :param reference_values_1d: 1-D numpy array of reference values -- i.e.,
        values from reference dataset at equally spaced quantile levels.
    :return: data_values: Same as input but in z-scores now.
    """

    # TODO(thunderhoser): Still need unit test.

    if numpy.all(numpy.isnan(reference_values_1d)):
        data_values[numpy.isfinite(data_values)] = 0.
        return data_values

    num_quantiles = len(reference_values_1d)
    quantile_levels = numpy.linspace(0, 1, num=num_quantiles, dtype=float)
    _, unique_indices = numpy.unique(reference_values_1d, return_index=True)

    interp_object = interp1d(
        x=reference_values_1d[unique_indices],
        y=quantile_levels[unique_indices],
        kind='linear',
        bounds_error=False,
        fill_value='extrapolate',
        assume_sorted=True
    )
    data_values = interp_object(data_values)

    # real_reference_values_1d = reference_values_1d[
    #     numpy.invert(numpy.isnan(reference_values_1d))
    # ]
    #
    # search_indices = numpy.searchsorted(
    #     a=numpy.sort(real_reference_values_1d), v=data_values, side='left'
    # ).astype(float)
    #
    # search_indices[numpy.invert(numpy.isfinite(data_values))] = numpy.nan
    # num_reference_vals = len(real_reference_values_1d)
    # data_values = search_indices / (num_reference_vals - 1)

    data_values = numpy.minimum(data_values, MAX_CUMULATIVE_DENSITY)
    data_values = numpy.maximum(data_values, MIN_CUMULATIVE_DENSITY)
    return scipy.stats.norm.ppf(data_values, loc=0., scale=1.)


def _quantile_denormalize_1var(data_values, reference_values_1d):
    """Does quantile *de*normalization for one variable.

    :param data_values: numpy array of data in z-score units.
    :param reference_values_1d: 1-D numpy array of reference values -- i.e.,
        values from reference dataset at equally spaced quantile levels.
    :return: data_values: Same as input but in physical units now.
    """

    # TODO(thunderhoser): Still need unit test.

    if numpy.all(numpy.isnan(reference_values_1d)):
        return data_values

    data_values = scipy.stats.norm.cdf(data_values, loc=0., scale=1.)
    real_reference_values_1d = reference_values_1d[
        numpy.invert(numpy.isnan(reference_values_1d))
    ]

    # Linear produces biased estimates (range of 0...0.1 in my test), while
    # lower produces unbiased estimates (range of -0.1...+0.1 in my test).
    real_flags = numpy.isfinite(data_values)
    data_values[real_flags] = numpy.percentile(
        numpy.ravel(real_reference_values_1d),
        100 * data_values[real_flags],
        interpolation='linear'
        # interpolation='lower'
    )

    return data_values


def get_intermediate_norm_params_for_nwp(
        interp_nwp_file_names, field_names, precip_forecast_hours,
        num_sample_values_per_file):
    """Computes intermediate normalization parameters for NWP data.

    'Final' normalization params = mean, stdev, and quantiles for every variable

    'Intermediate' normalization params = mean, mean of squares, number of
    values, and sample values for every variable.  Intermediate normalization
    params computed on many chunks of data can later be combined into final
    normalization params.

    :param interp_nwp_file_names: See doc for
        `get_normalization_params_for_nwp`.
    :param field_names: Same.
    :param precip_forecast_hours: Same.
    :param num_sample_values_per_file: Same.
    :return: intermediate_norm_param_table_xarray: xarray table with
        intermediate normalization parameters.  Metadata and variable names in
        this table should make it self-explanatory.
    """

    # Check input args.
    error_checking.assert_is_string_list(interp_nwp_file_names)
    error_checking.assert_is_string_list(field_names)
    error_checking.assert_is_geq(num_sample_values_per_file, 10)

    error_checking.assert_is_numpy_array(
        precip_forecast_hours, num_dimensions=1
    )
    error_checking.assert_is_integer_numpy_array(precip_forecast_hours)
    error_checking.assert_is_greater_numpy_array(precip_forecast_hours, 0)

    for this_field_name in field_names:
        nwp_model_utils.check_field_name(this_field_name)

    # Do actual stuff.
    norm_param_dict_dict = {}
    num_sample_values_total = (
        len(interp_nwp_file_names) * num_sample_values_per_file
    )

    for this_field_name in field_names:
        if this_field_name not in ACCUM_PRECIP_FIELD_NAMES:
            norm_param_dict_dict[this_field_name] = {
                NUM_VALUES_KEY: 0,
                MEAN_VALUE_KEY: 0.,
                MEAN_OF_SQUARES_KEY: 0.,
                SAMPLE_VALUES_KEY:
                    numpy.full(num_sample_values_total, numpy.nan)
            }
            continue

        for this_forecast_hour in precip_forecast_hours:
            norm_param_dict_dict[this_field_name, this_forecast_hour] = {
                NUM_VALUES_KEY: 0,
                MEAN_VALUE_KEY: 0.,
                MEAN_OF_SQUARES_KEY: 0.,
                SAMPLE_VALUES_KEY:
                    numpy.full(num_sample_values_total, numpy.nan)
            }

    for i in range(len(interp_nwp_file_names)):
        print('Reading data from: "{0:s}"...'.format(interp_nwp_file_names[i]))
        nwp_forecast_table_xarray = interp_nwp_model_io.read_file(
            interp_nwp_file_names[i]
        )

        nwpft = nwp_forecast_table_xarray
        nwp_field_names = nwpft.coords[nwp_model_utils.FIELD_DIM].values
        nwp_forecast_hours = numpy.round(
            nwpft.coords[nwp_model_utils.FORECAST_HOUR_DIM].values
        ).astype(int)

        for j in range(len(nwp_field_names)):
            f = nwp_field_names[j]
            if f not in field_names:
                continue

            if f not in ACCUM_PRECIP_FIELD_NAMES:
                norm_param_dict_dict[f] = _update_norm_params_1var_1file(
                    norm_param_dict=norm_param_dict_dict[f],
                    new_data_matrix=
                    nwpft[nwp_model_utils.DATA_KEY].values[..., j],
                    num_sample_values_per_file=num_sample_values_per_file,
                    file_index=i
                )
                continue

            for k in range(len(nwp_forecast_hours)):
                if nwp_forecast_hours[k] not in precip_forecast_hours:
                    continue

                h = nwp_forecast_hours[k]

                norm_param_dict_dict[f, h] = _update_norm_params_1var_1file(
                    norm_param_dict=norm_param_dict_dict[f, h],
                    new_data_matrix=
                    nwpft[nwp_model_utils.DATA_KEY].values[k, ..., j],
                    num_sample_values_per_file=num_sample_values_per_file,
                    file_index=i
                )

    num_fields = len(field_names)
    num_precip_hours = len(precip_forecast_hours)

    mean_value_matrix = numpy.full(
        (num_precip_hours, num_fields), numpy.nan
    )
    mean_squared_value_matrix = numpy.full(
        (num_precip_hours, num_fields), numpy.nan
    )
    num_values_matrix = numpy.full(
        (num_precip_hours, num_fields), -1, dtype=int
    )
    precip_sample_value_matrix = numpy.full(
        (num_precip_hours, num_sample_values_total), numpy.nan
    )
    nonprecip_sample_value_matrix = numpy.full(
        (num_fields, num_sample_values_total), numpy.nan
    )

    for j in range(num_fields):
        for k in range(num_precip_hours):
            f = field_names[j]
            h = precip_forecast_hours[k]

            if f in ACCUM_PRECIP_FIELD_NAMES:
                mean_value_matrix[k, j] = (
                    norm_param_dict_dict[f, h][MEAN_VALUE_KEY]
                )
                mean_squared_value_matrix[k, j] = (
                    norm_param_dict_dict[f, h][MEAN_OF_SQUARES_KEY]
                )
                num_values_matrix[k, j] = (
                    norm_param_dict_dict[f, h][NUM_VALUES_KEY]
                )
                precip_sample_value_matrix[k, :] = (
                    norm_param_dict_dict[f, h][SAMPLE_VALUES_KEY]
                )

                print((
                    'Mean, mean square, and num values for {0:s} at '
                    '{1:d}-hour lead = {2:.4g}, {3:.4g}, {4:d}'
                ).format(
                    field_names[j],
                    precip_forecast_hours[k],
                    mean_value_matrix[k, j],
                    mean_squared_value_matrix[k, j],
                    num_values_matrix[k, j]
                ))
            else:
                mean_value_matrix[k, j] = (
                    norm_param_dict_dict[f][MEAN_VALUE_KEY]
                )
                mean_squared_value_matrix[k, j] = (
                    norm_param_dict_dict[f][MEAN_OF_SQUARES_KEY]
                )
                num_values_matrix[k, j] = (
                    norm_param_dict_dict[f][NUM_VALUES_KEY]
                )

                if k > 0:
                    continue

                nonprecip_sample_value_matrix[j, :] = (
                    norm_param_dict_dict[f][SAMPLE_VALUES_KEY]
                )

                print((
                    'Mean, mean square, and num values for {0:s} = '
                    '{1:.4g}, {2:.4g}, {3:.4g}'
                ).format(
                    field_names[j],
                    mean_value_matrix[k, j],
                    mean_squared_value_matrix[k, j],
                    num_values_matrix[k, j]
                ))

    coord_dict = {
        nwp_model_utils.FORECAST_HOUR_DIM: precip_forecast_hours,
        nwp_model_utils.FIELD_DIM: field_names,
        nwp_model_utils.SAMPLE_VALUE_DIM: numpy.linspace(
            0, num_sample_values_total - 1,
            num=num_sample_values_total, dtype=int
        )
    }

    these_dim = (nwp_model_utils.FORECAST_HOUR_DIM, nwp_model_utils.FIELD_DIM)
    main_data_dict = {
        nwp_model_utils.MEAN_VALUE_KEY: (these_dim, mean_value_matrix),
        nwp_model_utils.MEAN_SQUARED_VALUE_KEY: (
            these_dim, mean_squared_value_matrix
        ),
        nwp_model_utils.NUM_VALUES_KEY: (
            these_dim, num_values_matrix.astype(float)
        )
    }

    these_dim = (
        nwp_model_utils.FORECAST_HOUR_DIM, nwp_model_utils.SAMPLE_VALUE_DIM
    )
    main_data_dict.update({
        nwp_model_utils.PRECIP_SAMPLE_VALUE_KEY: (
            these_dim, precip_sample_value_matrix
        )
    })

    these_dim = (
        nwp_model_utils.FIELD_DIM, nwp_model_utils.SAMPLE_VALUE_DIM
    )
    main_data_dict.update({
        nwp_model_utils.NONPRECIP_SAMPLE_VALUE_KEY: (
            these_dim, nonprecip_sample_value_matrix
        )
    })

    return xarray.Dataset(data_vars=main_data_dict, coords=coord_dict)


def intermediate_to_final_normalization_params(
        intermediate_norm_param_tables_xarray, num_quantiles):
    """Computes final normalization params from intermediate ones.

    Each set of intermediate normalization params is based on a different chunk
    of data -- e.g., one possibility is that each set of intermediate params is
    based on a different NWP model.

    :param intermediate_norm_param_tables_xarray: 1-D list of xarray tables with
        intermediate normalization params, each produced by
        `get_intermediate_norm_params_for_nwp`.
    :param num_quantiles: Number of quantiles to store for each variable.  The
        quantile levels will be evenly spaced from 0 to 1 (i.e., the 0th to
        100th percentile).
    :return: normalization_param_table_xarray: xarray table with final
        normalization parameters.  Metadata and variable names in this table
        should make it self-explanatory.
    """

    # Check input args.
    first_table = intermediate_norm_param_tables_xarray[0]

    for this_table in intermediate_norm_param_tables_xarray[1:]:
        assert numpy.array_equal(
            first_table.coords[nwp_model_utils.FORECAST_HOUR_DIM].values,
            this_table.coords[nwp_model_utils.FORECAST_HOUR_DIM].values
        )
        assert numpy.array_equal(
            first_table.coords[nwp_model_utils.FIELD_DIM].values,
            this_table.coords[nwp_model_utils.FIELD_DIM].values
        )

    error_checking.assert_is_geq(num_quantiles, 100)

    # Do actual stuff.
    precip_forecast_hours = (
        first_table.coords[nwp_model_utils.FORECAST_HOUR_DIM].values
    )
    field_names = first_table.coords[nwp_model_utils.FIELD_DIM].values

    num_fields = len(field_names)
    num_precip_hours = len(precip_forecast_hours)
    quantile_levels = numpy.linspace(0, 1, num=num_quantiles, dtype=float)

    mean_value_matrix = numpy.full(
        (num_precip_hours, num_fields), numpy.nan
    )
    mean_squared_value_matrix = numpy.full(
        (num_precip_hours, num_fields), numpy.nan
    )
    stdev_matrix = numpy.full(
        (num_precip_hours, num_fields), numpy.nan
    )
    quantile_matrix = numpy.full(
        (num_precip_hours, num_fields, num_quantiles), numpy.nan
    )

    for h in range(num_precip_hours):
        for f in range(num_fields):
            these_counts = numpy.array([
                t[nwp_model_utils.NUM_VALUES_KEY].values[h, f]
                for t in intermediate_norm_param_tables_xarray
            ])

            if numpy.sum(these_counts) < TOLERANCE:
                continue

            these_means = numpy.array([
                t[nwp_model_utils.MEAN_VALUE_KEY].values[h, f]
                for t in intermediate_norm_param_tables_xarray
            ])
            mean_value_matrix[h, f] = numpy.average(
                these_means, weights=these_counts
            )

            these_mean_squares = numpy.array([
                t[nwp_model_utils.MEAN_SQUARED_VALUE_KEY].values[h, f]
                for t in intermediate_norm_param_tables_xarray
            ])
            mean_squared_value_matrix[h, f] = numpy.average(
                these_mean_squares, weights=these_counts
            )

            this_norm_param_dict = {
                NUM_VALUES_KEY: numpy.sum(these_counts).astype(int),
                MEAN_VALUE_KEY: mean_value_matrix[h, f],
                MEAN_OF_SQUARES_KEY: mean_squared_value_matrix[h, f]
            }
            stdev_matrix[h, f] = _get_standard_deviation_1var(
                this_norm_param_dict
            )

            if field_names[f] in ACCUM_PRECIP_FIELD_NAMES:
                these_sample_values = numpy.concatenate([
                    t[nwp_model_utils.PRECIP_SAMPLE_VALUE_KEY].values[h, :]
                    for t in intermediate_norm_param_tables_xarray
                ])
            else:
                these_sample_values = numpy.concatenate([
                    t[nwp_model_utils.NONPRECIP_SAMPLE_VALUE_KEY].values[f, :]
                    for t in intermediate_norm_param_tables_xarray
                ])

            quantile_matrix[h, f, :] = numpy.nanpercentile(
                these_sample_values, 100 * quantile_levels
            )

            if field_names[f] in ACCUM_PRECIP_FIELD_NAMES:
                print((
                    'Mean, mean square, and standard deviation for {0:s} at '
                    '{1:d}-hour lead = {2:.4g}, {3:.4g}, {4:.4g}'
                ).format(
                    field_names[f],
                    precip_forecast_hours[h],
                    mean_value_matrix[h, f],
                    mean_squared_value_matrix[h, f],
                    stdev_matrix[h, f]
                ))
            else:
                if h > 0:
                    continue

                print((
                    'Mean, mean square, and standard deviation for {0:s} = '
                    '{1:.4g}, {2:.4g}, {3:.4g}'
                ).format(
                    field_names[f],
                    mean_value_matrix[h, f],
                    mean_squared_value_matrix[h, f],
                    stdev_matrix[h, f]
                ))

            for m in range(num_quantiles)[::10]:
                print((
                    '{0:.2f}th percentile for {1:s}{2:s} = {3:.4g}'
                ).format(
                    100 * quantile_levels[m],
                    field_names[f],
                    ' at {0:d}-hour lead'.format(precip_forecast_hours[h])
                    if field_names[f] in ACCUM_PRECIP_FIELD_NAMES
                    else '',
                    quantile_matrix[h, f, m]
                ))

    coord_dict = {
        nwp_model_utils.FORECAST_HOUR_DIM: precip_forecast_hours,
        nwp_model_utils.FIELD_DIM: field_names,
        nwp_model_utils.QUANTILE_LEVEL_DIM: quantile_levels
    }

    these_dim = (nwp_model_utils.FORECAST_HOUR_DIM, nwp_model_utils.FIELD_DIM)
    main_data_dict = {
        nwp_model_utils.MEAN_VALUE_KEY: (these_dim, mean_value_matrix),
        nwp_model_utils.MEAN_SQUARED_VALUE_KEY: (
            these_dim, mean_squared_value_matrix
        ),
        nwp_model_utils.STDEV_KEY: (these_dim, stdev_matrix)
    }

    these_dim = (
        nwp_model_utils.FORECAST_HOUR_DIM, nwp_model_utils.FIELD_DIM,
        nwp_model_utils.QUANTILE_LEVEL_DIM
    )
    main_data_dict.update({
        nwp_model_utils.QUANTILE_KEY: (these_dim, quantile_matrix)
    })

    return xarray.Dataset(data_vars=main_data_dict, coords=coord_dict)


def get_normalization_params_for_nwp(
        interp_nwp_file_names, field_names, precip_forecast_hours,
        num_quantiles, num_sample_values_per_file):
    """Computes normalization parameters for NWP data.

    This method computes both z-score parameters (mean and standard deviation),
    as well as quantiles, for each variable.  The z-score parameters are used
    for simple z-score normalization, and the quantiles are used for quantile
    normalization (which is always followed by conversion to the standard normal
    distribution, i.e., z-scores).

    :param interp_nwp_file_names: 1-D list of paths to input files (will be read
        by `interp_nwp_model_io.read_file`).
    :param field_names: 1-D list of fields for which to compute normalization
        params.
    :param precip_forecast_hours: 1-D numpy array of forecast hours at which to
        compute normalization params for precipitation.  For all other
        variables, normalization params will be time-independent.
    :param num_quantiles: Number of quantiles to store for each variable.  The
        quantile levels will be evenly spaced from 0 to 1 (i.e., the 0th to
        100th percentile).
    :param num_sample_values_per_file: Number of sample values per file to use
        for computing quantiles.  This value will be applied to each variable.
    :return: normalization_param_table_xarray: xarray table with normalization
        parameters.  Metadata and variable names in this table should make it
        self-explanatory.
    """

    # Check input args.
    error_checking.assert_is_string_list(interp_nwp_file_names)
    error_checking.assert_is_string_list(field_names)
    error_checking.assert_is_geq(num_sample_values_per_file, 10)
    error_checking.assert_is_geq(num_quantiles, 100)

    error_checking.assert_is_numpy_array(
        precip_forecast_hours, num_dimensions=1
    )
    error_checking.assert_is_integer_numpy_array(precip_forecast_hours)
    error_checking.assert_is_greater_numpy_array(precip_forecast_hours, 0)

    for this_field_name in field_names:
        nwp_model_utils.check_field_name(this_field_name)

    # Do actual stuff.
    norm_param_dict_dict = {}
    num_sample_values_total = (
        len(interp_nwp_file_names) * num_sample_values_per_file
    )

    for this_field_name in field_names:
        if this_field_name not in ACCUM_PRECIP_FIELD_NAMES:
            norm_param_dict_dict[this_field_name] = {
                NUM_VALUES_KEY: 0,
                MEAN_VALUE_KEY: 0.,
                MEAN_OF_SQUARES_KEY: 0.,
                SAMPLE_VALUES_KEY:
                    numpy.full(num_sample_values_total, numpy.nan)
            }
            continue

        for this_forecast_hour in precip_forecast_hours:
            norm_param_dict_dict[this_field_name, this_forecast_hour] = {
                NUM_VALUES_KEY: 0,
                MEAN_VALUE_KEY: 0.,
                MEAN_OF_SQUARES_KEY: 0.,
                SAMPLE_VALUES_KEY:
                    numpy.full(num_sample_values_total, numpy.nan)
            }

    for i in range(len(interp_nwp_file_names)):
        print('Reading data from: "{0:s}"...'.format(interp_nwp_file_names[i]))
        nwp_forecast_table_xarray = interp_nwp_model_io.read_file(
            interp_nwp_file_names[i]
        )

        nwpft = nwp_forecast_table_xarray
        nwp_field_names = nwpft.coords[nwp_model_utils.FIELD_DIM].values
        nwp_forecast_hours = numpy.round(
            nwpft.coords[nwp_model_utils.FORECAST_HOUR_DIM].values
        ).astype(int)

        for j in range(len(nwp_field_names)):
            f = nwp_field_names[j]
            if f not in field_names:
                continue

            if f not in ACCUM_PRECIP_FIELD_NAMES:
                norm_param_dict_dict[f] = _update_norm_params_1var_1file(
                    norm_param_dict=norm_param_dict_dict[f],
                    new_data_matrix=
                    nwpft[nwp_model_utils.DATA_KEY].values[..., j],
                    num_sample_values_per_file=num_sample_values_per_file,
                    file_index=i
                )
                continue

            for k in range(len(nwp_forecast_hours)):
                if nwp_forecast_hours[k] not in precip_forecast_hours:
                    continue

                h = nwp_forecast_hours[k]

                norm_param_dict_dict[f, h] = _update_norm_params_1var_1file(
                    norm_param_dict=norm_param_dict_dict[f, h],
                    new_data_matrix=
                    nwpft[nwp_model_utils.DATA_KEY].values[k, ..., j],
                    num_sample_values_per_file=num_sample_values_per_file,
                    file_index=i
                )

    num_fields = len(field_names)
    num_precip_hours = len(precip_forecast_hours)
    quantile_levels = numpy.linspace(0, 1, num=num_quantiles, dtype=float)

    mean_value_matrix = numpy.full(
        (num_precip_hours, num_fields), numpy.nan
    )
    mean_squared_value_matrix = numpy.full(
        (num_precip_hours, num_fields), numpy.nan
    )
    stdev_matrix = numpy.full(
        (num_precip_hours, num_fields), numpy.nan
    )
    quantile_matrix = numpy.full(
        (num_precip_hours, num_fields, num_quantiles), numpy.nan
    )

    for j in range(num_fields):
        for k in range(num_precip_hours):
            f = field_names[j]
            h = precip_forecast_hours[k]

            if f in ACCUM_PRECIP_FIELD_NAMES:
                mean_value_matrix[k, j] = (
                    norm_param_dict_dict[f, h][MEAN_VALUE_KEY]
                )
                mean_squared_value_matrix[k, j] = (
                    norm_param_dict_dict[f, h][MEAN_OF_SQUARES_KEY]
                )
                stdev_matrix[k, j] = _get_standard_deviation_1var(
                    norm_param_dict_dict[f, h]
                )
                quantile_matrix[k, j, :] = numpy.nanpercentile(
                    norm_param_dict_dict[f, h][SAMPLE_VALUES_KEY],
                    100 * quantile_levels
                )

                print((
                    'Mean, mean square, and standard deviation for {0:s} at '
                    '{1:d}-hour lead = {2:.4g}, {3:.4g}, {4:.4g}'
                ).format(
                    field_names[j],
                    precip_forecast_hours[k],
                    mean_value_matrix[k, j],
                    mean_squared_value_matrix[k, j],
                    stdev_matrix[k, j]
                ))
            else:
                mean_value_matrix[k, j] = (
                    norm_param_dict_dict[f][MEAN_VALUE_KEY]
                )
                mean_squared_value_matrix[k, j] = (
                    norm_param_dict_dict[f][MEAN_OF_SQUARES_KEY]
                )
                stdev_matrix[k, j] = _get_standard_deviation_1var(
                    norm_param_dict_dict[f]
                )
                quantile_matrix[k, j, :] = numpy.nanpercentile(
                    norm_param_dict_dict[f][SAMPLE_VALUES_KEY],
                    100 * quantile_levels
                )

                if k > 0:
                    continue

                print((
                    'Mean, mean square, and standard deviation for {0:s} = '
                    '{1:.4g}, {2:.4g}, {3:.4g}'
                ).format(
                    field_names[j],
                    mean_value_matrix[k, j],
                    mean_squared_value_matrix[k, j],
                    stdev_matrix[k, j]
                ))

            for m in range(num_quantiles)[::10]:
                print((
                    '{0:.2f}th percentile for {1:s}{2:s} = {3:.4g}'
                ).format(
                    100 * quantile_levels[m],
                    field_names[j],
                    ' at {0:d}-hour lead'.format(precip_forecast_hours[k])
                    if f in ACCUM_PRECIP_FIELD_NAMES
                    else '',
                    quantile_matrix[k, j, m]
                ))

    coord_dict = {
        nwp_model_utils.FORECAST_HOUR_DIM: precip_forecast_hours,
        nwp_model_utils.FIELD_DIM: field_names,
        nwp_model_utils.QUANTILE_LEVEL_DIM: quantile_levels
    }

    these_dim = (nwp_model_utils.FORECAST_HOUR_DIM, nwp_model_utils.FIELD_DIM)
    main_data_dict = {
        nwp_model_utils.MEAN_VALUE_KEY: (these_dim, mean_value_matrix),
        nwp_model_utils.MEAN_SQUARED_VALUE_KEY: (
            these_dim, mean_squared_value_matrix
        ),
        nwp_model_utils.STDEV_KEY: (these_dim, stdev_matrix)
    }

    these_dim = (
        nwp_model_utils.FORECAST_HOUR_DIM, nwp_model_utils.FIELD_DIM,
        nwp_model_utils.QUANTILE_LEVEL_DIM
    )
    main_data_dict.update({
        nwp_model_utils.QUANTILE_KEY: (these_dim, quantile_matrix)
    })

    return xarray.Dataset(data_vars=main_data_dict, coords=coord_dict)


def normalize_nwp_data(nwp_forecast_table_xarray, norm_param_table_xarray,
                       use_quantile_norm):
    """Normalizes NWP data.

    :param nwp_forecast_table_xarray: xarray table with NWP data in physical
        units.
    :param norm_param_table_xarray: xarray table with normalization
        parameters, created by `get_normalization_params_for_nwp`.
    :param use_quantile_norm: Boolean flag.  If True, will use quantile
        normalization and then convert ranks to standard normal distribution.
        If False, will just use z-score normalization.
    :return: nwp_forecast_table_xarray: Same as input but normalized.
    """

    nwpft = nwp_forecast_table_xarray
    npt = norm_param_table_xarray
    error_checking.assert_is_boolean(use_quantile_norm)

    field_names = nwpft.coords[nwp_model_utils.FIELD_DIM].values.tolist()
    forecast_hours = numpy.round(
        nwpft.coords[nwp_model_utils.FORECAST_HOUR_DIM].values
    ).astype(int)

    num_forecast_hours = len(forecast_hours)
    num_fields = len(field_names)
    data_matrix = nwpft[nwp_model_utils.DATA_KEY].values

    for j in range(num_fields):
        j_new = numpy.where(
            npt.coords[nwp_model_utils.FIELD_DIM].values == field_names[j]
        )[0][0]

        if field_names[j] not in ACCUM_PRECIP_FIELD_NAMES:
            if use_quantile_norm:
                data_matrix[..., j] = _quantile_normalize_1var(
                    data_values=data_matrix[..., j],
                    reference_values_1d=
                    npt[nwp_model_utils.QUANTILE_KEY].values[0, j_new, :]
                )
            else:
                data_matrix[..., j] = _z_normalize_1var(
                    data_values=data_matrix[..., j],
                    reference_mean=
                    npt[nwp_model_utils.MEAN_VALUE_KEY].values[0, j_new],
                    reference_stdev=
                    npt[nwp_model_utils.STDEV_KEY].values[0, j_new]
                )

            continue

        for k in range(num_forecast_hours):
            k_new = numpy.where(
                numpy.round(
                    npt.coords[nwp_model_utils.FORECAST_HOUR_DIM].values
                ).astype(int)
                == forecast_hours[k]
            )[0][0]

            if use_quantile_norm:
                data_matrix[k, ..., j] = _quantile_normalize_1var(
                    data_values=data_matrix[k, ..., j],
                    reference_values_1d=
                    npt[nwp_model_utils.QUANTILE_KEY].values[k_new, j_new, :]
                )
            else:
                data_matrix[k, ..., j] = _z_normalize_1var(
                    data_values=data_matrix[k, ..., j],
                    reference_mean=
                    npt[nwp_model_utils.MEAN_VALUE_KEY].values[k_new, j_new],
                    reference_stdev=
                    npt[nwp_model_utils.STDEV_KEY].values[k_new, j_new]
                )

    return nwp_forecast_table_xarray.assign({
        nwp_model_utils.DATA_KEY: (
            nwp_forecast_table_xarray[nwp_model_utils.DATA_KEY].dims,
            data_matrix
        )
    })


def denormalize_nwp_data(nwp_forecast_table_xarray, norm_param_table_xarray,
                         use_quantile_norm):
    """Denormalizes NWP data.

    :param nwp_forecast_table_xarray: xarray table with NWP data in z-scores.
    :param norm_param_table_xarray: xarray table with normalization
        parameters, created by `get_normalization_params_for_nwp`.
    :param use_quantile_norm: Boolean flag.  If True, will assume that
        normalization method was quantile normalization followed by converting
        ranks to standard normal distribution.  If False, will assume just
        z-score normalization.
    :return: nwp_forecast_table_xarray: Same as input but in physical units.
    """

    nwpft = nwp_forecast_table_xarray
    npt = norm_param_table_xarray
    error_checking.assert_is_boolean(use_quantile_norm)

    field_names = nwpft.coords[nwp_model_utils.FIELD_DIM].values.tolist()
    forecast_hours = numpy.round(
        nwpft.coords[nwp_model_utils.FORECAST_HOUR_DIM].values
    ).astype(int)

    num_forecast_hours = len(forecast_hours)
    num_fields = len(field_names)
    data_matrix = nwpft[nwp_model_utils.DATA_KEY].values

    for j in range(num_fields):
        j_new = numpy.where(
            npt.coords[nwp_model_utils.FIELD_DIM].values == field_names[j]
        )[0][0]

        if field_names[j] not in ACCUM_PRECIP_FIELD_NAMES:
            if use_quantile_norm:
                data_matrix[..., j] = _quantile_denormalize_1var(
                    data_values=data_matrix[..., j],
                    reference_values_1d=
                    npt[nwp_model_utils.QUANTILE_KEY].values[0, j_new, :]
                )
            else:
                data_matrix[..., j] = _z_denormalize_1var(
                    data_values=data_matrix[..., j],
                    reference_mean=
                    npt[nwp_model_utils.MEAN_VALUE_KEY].values[0, j_new],
                    reference_stdev=
                    npt[nwp_model_utils.STDEV_KEY].values[0, j_new]
                )

            continue

        for k in range(num_forecast_hours):
            k_new = numpy.where(
                numpy.round(
                    npt.coords[nwp_model_utils.FORECAST_HOUR_DIM].values
                ).astype(int)
                == forecast_hours[k]
            )[0][0]

            if use_quantile_norm:
                data_matrix[k, ..., j] = _quantile_denormalize_1var(
                    data_values=data_matrix[k, ..., j],
                    reference_values_1d=
                    npt[nwp_model_utils.QUANTILE_KEY].values[k_new, j_new, :]
                )
            else:
                data_matrix[k, ..., j] = _z_denormalize_1var(
                    data_values=data_matrix[k, ..., j],
                    reference_mean=
                    npt[nwp_model_utils.MEAN_VALUE_KEY].values[k_new, j_new],
                    reference_stdev=
                    npt[nwp_model_utils.STDEV_KEY].values[k_new, j_new]
                )

    return nwp_forecast_table_xarray.assign({
        nwp_model_utils.DATA_KEY: (
            nwp_forecast_table_xarray[nwp_model_utils.DATA_KEY].dims,
            data_matrix
        )
    })


def get_normalization_params_for_targets(
        urma_file_names, num_quantiles, num_sample_values_per_file):
    """Computes normalization params for each URMA target variable.

    :param urma_file_names: 1-D list of paths to URMA files (will be read by
        `urma_io.read_file`).
    :param num_quantiles: See documentation for
        `get_normalization_params_for_nwp`.
    :param num_sample_values_per_file: Same.
    :return: normalization_param_table_xarray: xarray table with normalization
        parameters.  Metadata and variable names in this table should make it
        self-explanatory.
    """

    # Check input args.
    error_checking.assert_is_string_list(urma_file_names)
    error_checking.assert_is_geq(num_sample_values_per_file, 10)
    error_checking.assert_is_geq(num_quantiles, 100)

    # Housekeeping.
    first_urma_table_xarray = urma_io.read_file(urma_file_names[0])
    field_names = (
        first_urma_table_xarray.coords[urma_utils.FIELD_DIM].values.tolist()
    )

    norm_param_dict_dict = {}
    num_sample_values_total = len(urma_file_names) * num_sample_values_per_file

    for this_field_name in field_names:
        norm_param_dict_dict[this_field_name] = {
            NUM_VALUES_KEY: 0,
            MEAN_VALUE_KEY: 0.,
            MEAN_OF_SQUARES_KEY: 0.,
            SAMPLE_VALUES_KEY: numpy.full(num_sample_values_total, numpy.nan)
        }

    quantile_levels = numpy.linspace(0, 1, num=num_quantiles, dtype=float)

    # Do actual stuff.
    for i in range(len(urma_file_names)):
        print('Reading data from: "{0:s}"...'.format(urma_file_names[i]))
        this_urma_table_xarray = urma_io.read_file(urma_file_names[i])
        tutx = this_urma_table_xarray

        for j in range(len(tutx.coords[urma_utils.FIELD_DIM].values)):
            f = tutx.coords[urma_utils.FIELD_DIM].values[j]

            norm_param_dict_dict[f] = _update_norm_params_1var_1file(
                norm_param_dict=norm_param_dict_dict[f],
                new_data_matrix=tutx[urma_utils.DATA_KEY].values[..., j],
                num_sample_values_per_file=num_sample_values_per_file,
                file_index=i
            )

    num_fields = len(field_names)
    mean_values = numpy.full(num_fields, numpy.nan)
    mean_squared_values = numpy.full(num_fields, numpy.nan)
    stdev_values = numpy.full(num_fields, numpy.nan)
    quantile_matrix = numpy.full((num_fields, num_quantiles), numpy.nan)

    for j in range(num_fields):
        f = field_names[j]

        mean_values[j] = norm_param_dict_dict[f][MEAN_VALUE_KEY]
        mean_squared_values[j] = norm_param_dict_dict[f][MEAN_OF_SQUARES_KEY]
        stdev_values[j] = _get_standard_deviation_1var(norm_param_dict_dict[f])
        quantile_matrix[j, :] = numpy.nanpercentile(
            norm_param_dict_dict[f][SAMPLE_VALUES_KEY],
            100 * quantile_levels
        )

        print((
            'Mean, mean square, and standard deviation for {0:s} = '
            '{1:.4g}, {2:.4g}, {3:.4g}'
        ).format(
            field_names[j],
            mean_values[j], mean_squared_values[j], stdev_values[j]
        ))

        for m in range(num_quantiles)[::10]:
            print('{0:.2f}th percentile for {1:s} = {2:.4g}'.format(
                100 * quantile_levels[m],
                field_names[j],
                quantile_matrix[j, m]
            ))

    coord_dict = {
        urma_utils.FIELD_DIM: field_names,
        urma_utils.QUANTILE_LEVEL_DIM: quantile_levels
    }

    these_dim = (urma_utils.FIELD_DIM,)
    main_data_dict = {
        urma_utils.MEAN_VALUE_KEY: (these_dim, mean_values),
        urma_utils.MEAN_SQUARED_VALUE_KEY: (these_dim, mean_squared_values),
        urma_utils.STDEV_KEY: (these_dim, stdev_values)
    }

    these_dim = (urma_utils.FIELD_DIM, urma_utils.QUANTILE_LEVEL_DIM)
    main_data_dict.update({
        urma_utils.QUANTILE_KEY: (these_dim, quantile_matrix)
    })

    return xarray.Dataset(data_vars=main_data_dict, coords=coord_dict)


def normalize_targets(urma_table_xarray, norm_param_table_xarray,
                      use_quantile_norm):
    """Normalizes target variables from physical units to z-scores.

    :param urma_table_xarray: xarray table with URMA data in physical units.
    :param norm_param_table_xarray: xarray table with normalization
        parameters, created by `get_normalization_params_for_targets`.
    :param use_quantile_norm: Boolean flag.  If True, will use quantile
        normalization and then convert ranks to standard normal distribution.
        If False, will just use z-score normalization.
    :return: urma_table_xarray: Same as input but normalized.
    """

    urmat = urma_table_xarray
    npt = norm_param_table_xarray

    field_names = urmat.coords[urma_utils.FIELD_DIM].values.tolist()
    num_fields = len(field_names)

    data_matrix = urmat[urma_utils.DATA_KEY].values

    for j in range(num_fields):
        j_new = numpy.where(
            npt.coords[urma_utils.FIELD_DIM].values == field_names[j]
        )[0][0]

        if use_quantile_norm:
            data_matrix[..., j] = _quantile_normalize_1var(
                data_values=data_matrix[..., j],
                reference_values_1d=
                npt[urma_utils.QUANTILE_KEY].values[j_new, :]
            )
        else:
            data_matrix[..., j] = _z_normalize_1var(
                data_values=data_matrix[..., j],
                reference_mean=npt[urma_utils.MEAN_VALUE_KEY].values[j_new],
                reference_stdev=npt[urma_utils.STDEV_KEY].values[j_new]
            )

    return urma_table_xarray.assign({
        urma_utils.DATA_KEY: (
            urma_table_xarray[urma_utils.DATA_KEY].dims, data_matrix
        )
    })


def denormalize_targets(urma_table_xarray, norm_param_table_xarray,
                        use_quantile_norm):
    """Denormalizes target variables from z-scores to physical units.

    :param urma_table_xarray: xarray table with URMA data in z-scores.
    :param norm_param_table_xarray: xarray table with normalization
        parameters, created by `get_normalization_params_for_targets`.
    :param use_quantile_norm: Boolean flag.  If True, will assume that
        normalization method was quantile normalization followed by converting
        ranks to standard normal distribution.  If False, will assume just
        z-score normalization.
    :return: urma_table_xarray: Same as input but in physical units.
    """

    urmat = urma_table_xarray
    npt = norm_param_table_xarray

    field_names = urmat.coords[urma_utils.FIELD_DIM].values.tolist()
    num_fields = len(field_names)

    data_matrix = urmat[urma_utils.DATA_KEY].values

    for j in range(num_fields):
        j_new = numpy.where(
            npt.coords[urma_utils.FIELD_DIM].values == field_names[j]
        )[0][0]

        if use_quantile_norm:
            data_matrix[..., j] = _quantile_denormalize_1var(
                data_values=data_matrix[..., j],
                reference_values_1d=
                npt[urma_utils.QUANTILE_KEY].values[j_new, :]
            )
        else:
            data_matrix[..., j] = _z_denormalize_1var(
                data_values=data_matrix[..., j],
                reference_mean=npt[urma_utils.MEAN_VALUE_KEY].values[j_new],
                reference_stdev=npt[urma_utils.STDEV_KEY].values[j_new]
            )

    return urma_table_xarray.assign({
        urma_utils.DATA_KEY: (
            urma_table_xarray[urma_utils.DATA_KEY].dims, data_matrix
        )
    })


def get_normalization_params_for_nbm_const(nbm_constant_file_name,
                                           num_quantiles):
    """Computes normalization parameters for each NBM time-constant variable.

    :param nbm_constant_file_name: Path to input file (will be read by
        `nbm_constant_io.read_file`).
    :param num_quantiles: See doc for `get_normalization_params_for_nwp`.
    :return: norm_param_table_xarray: Same.
    """

    error_checking.assert_is_geq(num_quantiles, 100)

    print('Reading data from: "{0:s}"...'.format(nbm_constant_file_name))
    nbm_constant_table_xarray = nbm_constant_io.read_file(
        nbm_constant_file_name
    )
    nbmct = nbm_constant_table_xarray

    field_names = nbmct.coords[nbm_constant_utils.FIELD_DIM].values.tolist()
    if nbm_constant_utils.LAND_SEA_MASK_NAME in field_names:
        field_names.remove(nbm_constant_utils.LAND_SEA_MASK_NAME)

    num_sample_values_total = (
        nbmct[nbm_constant_utils.DATA_KEY].values[..., 0].size
    )

    norm_param_dict_dict = {}
    for this_field_name in field_names:
        norm_param_dict_dict[this_field_name] = {
            NUM_VALUES_KEY: 0,
            MEAN_VALUE_KEY: 0.,
            MEAN_OF_SQUARES_KEY: 0.,
            SAMPLE_VALUES_KEY: numpy.full(num_sample_values_total, numpy.nan)
        }

    quantile_levels = numpy.linspace(0, 1, num=num_quantiles, dtype=float)

    for j in range(len(field_names)):
        j_new = numpy.where(
            nbmct.coords[nbm_constant_utils.FIELD_DIM].values == field_names[j]
        )[0][0]

        norm_param_dict_dict[field_names[j]] = _update_norm_params_1var_1file(
            norm_param_dict=norm_param_dict_dict[field_names[j]],
            new_data_matrix=
            nbmct[nbm_constant_utils.DATA_KEY].values[..., j_new],
            num_sample_values_per_file=num_sample_values_total,
            file_index=0
        )

    num_fields = len(field_names)
    mean_values = numpy.full(num_fields, numpy.nan)
    mean_squared_values = numpy.full(num_fields, numpy.nan)
    stdev_values = numpy.full(num_fields, numpy.nan)
    quantile_matrix = numpy.full((num_fields, num_quantiles), numpy.nan)

    for j in range(num_fields):
        f = field_names[j]

        mean_values[j] = norm_param_dict_dict[f][MEAN_VALUE_KEY]
        mean_squared_values[j] = norm_param_dict_dict[f][MEAN_OF_SQUARES_KEY]
        stdev_values[j] = _get_standard_deviation_1var(norm_param_dict_dict[f])
        quantile_matrix[j, :] = numpy.nanpercentile(
            norm_param_dict_dict[f][SAMPLE_VALUES_KEY],
            100 * quantile_levels
        )

        print((
            'Mean, squared mean, and standard deviation for {0:s} = '
            '{1:.4g}, {2:.4g}, {3:.4g}'
        ).format(
            field_names[j],
            mean_values[j], mean_squared_values[j], stdev_values[j]
        ))

        for m in range(num_quantiles)[::10]:
            print('{0:.2f}th percentile for {1:s} = {2:.4g}'.format(
                100 * quantile_levels[m],
                field_names[j],
                quantile_matrix[j, m]
            ))

    coord_dict = {
        nbm_constant_utils.FIELD_DIM: field_names,
        nbm_constant_utils.QUANTILE_LEVEL_DIM: quantile_levels
    }

    these_dim_1d = (nbm_constant_utils.FIELD_DIM,)
    these_dim_2d = (
        nbm_constant_utils.FIELD_DIM, nbm_constant_utils.QUANTILE_LEVEL_DIM
    )

    main_data_dict = {
        nbm_constant_utils.MEAN_VALUE_KEY: (these_dim_1d, mean_values),
        nbm_constant_utils.MEAN_SQUARED_VALUE_KEY: (
            these_dim_1d, mean_squared_values
        ),
        nbm_constant_utils.STDEV_KEY: (these_dim_1d, stdev_values),
        nbm_constant_utils.QUANTILE_KEY: (these_dim_2d, quantile_matrix)
    }

    return xarray.Dataset(data_vars=main_data_dict, coords=coord_dict)


def normalize_nbm_constants(
        nbm_constant_table_xarray, norm_param_table_xarray, use_quantile_norm):
    """Normalizes NBM time-constant variables.

    :param nbm_constant_table_xarray: xarray table with NBM constants in
        physical units.
    :param norm_param_table_xarray: xarray table with normalization
        parameters, created by `get_normalization_params_for_nbm_const`.
    :param use_quantile_norm: Boolean flag.  If True, will use quantile
        normalization and then convert ranks to standard normal distribution.
        If False, will just use z-score normalization.
    :return: nbm_constant_table_xarray: Same as input but normalized.
    """

    nbmct = nbm_constant_table_xarray
    npt = norm_param_table_xarray

    field_names = nbmct.coords[nbm_constant_utils.FIELD_DIM].values.tolist()
    num_fields = len(field_names)

    data_matrix = nbmct[nbm_constant_utils.DATA_KEY].values

    for j in range(num_fields):
        if field_names[j] == nbm_constant_utils.LAND_SEA_MASK_NAME:
            continue

        j_new = numpy.where(
            npt.coords[nbm_constant_utils.FIELD_DIM].values == field_names[j]
        )[0][0]

        if use_quantile_norm:
            data_matrix[..., j] = _quantile_normalize_1var(
                data_values=data_matrix[..., j],
                reference_values_1d=
                npt[nbm_constant_utils.QUANTILE_KEY].values[j_new, :]
            )
        else:
            data_matrix[..., j] = _z_normalize_1var(
                data_values=data_matrix[..., j],
                reference_mean=
                npt[nbm_constant_utils.MEAN_VALUE_KEY].values[j_new],
                reference_stdev=
                npt[nbm_constant_utils.STDEV_KEY].values[j_new]
            )

    return nbm_constant_table_xarray.assign({
        nbm_constant_utils.DATA_KEY: (
            nbm_constant_table_xarray[nbm_constant_utils.DATA_KEY].dims,
            data_matrix
        )
    })

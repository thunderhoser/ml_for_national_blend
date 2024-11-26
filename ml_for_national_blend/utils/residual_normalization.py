"""Helper methods for residual normalization."""

import os
import warnings
import numpy
import xarray
from ml_for_national_blend.outside_code import error_checking
from ml_for_national_blend.io import urma_io
from ml_for_national_blend.io import nwp_model_io
from ml_for_national_blend.io import interp_nwp_model_io
from ml_for_national_blend.utils import urma_utils
from ml_for_national_blend.utils import nwp_model_utils
from ml_for_national_blend.utils import normalization as non_resid_normalization

TOLERANCE = 1e-6
SECONDS_TO_HOURS = 1. / 3600

MIN_CUMULATIVE_DENSITY = 1e-6
MAX_CUMULATIVE_DENSITY = 1. - 1e-6
# MAX_CUMULATIVE_DENSITY = 0.9995  # To account for 16-bit floats.

NUM_VALUES_KEY = 'num_values'
MEAN_VALUE_KEY = 'mean_value'
MEAN_OF_SQUARES_KEY = 'mean_of_squares'

ACCUM_PRECIP_FIELD_NAMES = [nwp_model_utils.PRECIP_NAME]


def _update_norm_params_1var_1file(norm_param_dict, new_data_matrix):
    """Updates normalization params for one variable, based on one file.

    :param norm_param_dict: Dictionary with the following keys.
    norm_param_dict['num_values']: Number of values on which current
        estimates are based.
    norm_param_dict['mean_value']: Current mean.
    norm_param_dict['mean_of_squares']: Current mean of squared values.

    :param new_data_matrix: numpy array with new values.  Will be used to
        update estimates in `norm_param_dict`.
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


def _normalize_1var(z_score_values, reference_temporal_stdev):
    """Does residual normalization for one variable.

    :param z_score_values: numpy array of data in z-score units.
    :param reference_temporal_stdev: Temporal standard deviation of z-scores
        from reference (e.g., training) dataset.
    :return: residual_values: Same as input but in residual scores now.
    """

    if numpy.isnan(reference_temporal_stdev):
        return numpy.zeros_like(z_score_values)

    return z_score_values / reference_temporal_stdev


def _denormalize_1var(residual_values, reference_temporal_stdev):
    """Undoes residual normalization for one variable.

    :param residual_values: numpy array of data in residual units.
    :param reference_temporal_stdev: Temporal standard deviation of z-scores
        from reference (e.g., training) dataset.
    :return: z_score_values: Same as input but in z-scores now.
    """

    if numpy.isnan(reference_temporal_stdev):
        return numpy.zeros_like(residual_values)

    return residual_values * reference_temporal_stdev


def get_intermediate_norm_params_for_nwp(
        interp_nwp_file_names, non_resid_normalization_file_name,
        field_names, precip_forecast_hours):
    """Computes intermediate normalization parameters for NWP data.

    'Final' normalization params = temporal stdev for every variable

    'Intermediate' normalization params = mean temporal difference, mean squared
    temporal difference, and number of values for every variable.  Intermediate
    normalization params computed on many chunks of data can later be combined
    into final normalization params.

    :param interp_nwp_file_names: See doc for
        `get_normalization_params_for_nwp`.
    :param non_resid_normalization_file_name: Same.
    :param field_names: Same.
    :param precip_forecast_hours: Same.
    :return: intermediate_norm_param_table_xarray: xarray table with
        intermediate normalization parameters.  Metadata and variable names in
        this table should make it self-explanatory.
    """

    # Check input args.
    error_checking.assert_is_string_list(interp_nwp_file_names)
    error_checking.assert_is_string_list(field_names)

    error_checking.assert_is_numpy_array(
        precip_forecast_hours, num_dimensions=1
    )
    error_checking.assert_is_integer_numpy_array(precip_forecast_hours)
    error_checking.assert_is_greater_numpy_array(precip_forecast_hours, 0)

    for this_field_name in field_names:
        nwp_model_utils.check_field_name(this_field_name)

    # Do actual stuff.
    norm_param_dict_dict = {}

    for this_field_name in field_names:
        if this_field_name not in ACCUM_PRECIP_FIELD_NAMES:
            norm_param_dict_dict[this_field_name] = {
                NUM_VALUES_KEY: 0,
                MEAN_VALUE_KEY: 0.,
                MEAN_OF_SQUARES_KEY: 0.
            }
            continue

        for this_forecast_hour in precip_forecast_hours:
            norm_param_dict_dict[this_field_name, this_forecast_hour] = {
                NUM_VALUES_KEY: 0,
                MEAN_VALUE_KEY: 0.,
                MEAN_OF_SQUARES_KEY: 0.
            }

    print((
        'Reading params for non-residual normalization from: "{0:s}"...'
    ).format(
        non_resid_normalization_file_name
    ))
    non_resid_norm_param_table_xarray = nwp_model_io.read_normalization_file(
        non_resid_normalization_file_name
    )

    already_used_file_names = []

    for i in range(len(interp_nwp_file_names)):
        if interp_nwp_file_names[i] in already_used_file_names:
            continue

        this_model_name = interp_nwp_model_io.file_name_to_model_name(
            interp_nwp_file_names[i]
        )
        this_init_time_unix_sec = interp_nwp_model_io.file_name_to_init_time(
            interp_nwp_file_names[i]
        )
        these_forecast_hours = nwp_model_utils.model_to_forecast_hours(
            model_name=this_model_name,
            init_time_unix_sec=this_init_time_unix_sec
        )
        this_directory_name = '/'.join(interp_nwp_file_names[i].split('/')[:-2])

        these_file_names = [
            interp_nwp_model_io.find_file(
                directory_name=this_directory_name,
                init_time_unix_sec=this_init_time_unix_sec,
                forecast_hour=f,
                model_name=this_model_name,
                raise_error_if_missing=False
            )
            for f in these_forecast_hours
        ]
        these_file_names = [fn for fn in these_file_names if os.path.isfile(fn)]
        already_used_file_names += these_file_names

        nwp_forecast_tables_xarray = []
        for this_file_name in these_file_names:
            print('Reading data from: "{0:s}"...'.format(this_file_name))
            nwp_forecast_tables_xarray.append(
                interp_nwp_model_io.read_file(this_file_name)
            )

        nwp_forecast_tables_xarray = [
            nwpft.drop_vars(attrs=list(nwpft.attrs.keys()))
            for nwpft in nwp_forecast_tables_xarray
        ]

        try:
            nwp_forecast_table_xarray = xarray.concat(
                nwp_forecast_tables_xarray,
                dim=nwp_model_utils.FORECAST_HOUR_DIM,
                data_vars=[nwp_model_utils.DATA_KEY],
                coords='minimal', compat='identical', join='exact'
            )
        except Exception as this_exception:
            warning_string = (
                'POTENTIAL ERROR: Could not concatenate tables:\n{0:s}'
            ).format(
                str(this_exception)
            )

            warnings.warn(warning_string)
            continue

        del nwp_forecast_tables_xarray

        nwp_forecast_table_xarray = non_resid_normalization.normalize_nwp_data(
            nwp_forecast_table_xarray=nwp_forecast_table_xarray,
            norm_param_table_xarray=non_resid_norm_param_table_xarray,
            use_quantile_norm=True
        )

        nwpft = nwp_forecast_table_xarray
        nwp_field_names = nwpft.coords[nwp_model_utils.FIELD_DIM].values
        nwp_forecast_hours = numpy.round(
            nwpft.coords[nwp_model_utils.FORECAST_HOUR_DIM].values
        ).astype(int)

        nwp_forecast_hour_diffs = numpy.diff(nwp_forecast_hours)
        nwp_forecast_hour_diffs = numpy.concatenate([
            nwp_forecast_hour_diffs[[0]], nwp_forecast_hour_diffs
        ])
        nwp_forecast_hour_diffs = numpy.expand_dims(
            nwp_forecast_hour_diffs, axis=-1
        )
        nwp_forecast_hour_diffs = numpy.expand_dims(
            nwp_forecast_hour_diffs, axis=-1
        )

        for j in range(len(nwp_field_names)):
            f = nwp_field_names[j]
            if f not in field_names:
                continue

            if f not in ACCUM_PRECIP_FIELD_NAMES:
                this_diff_matrix = numpy.diff(
                    nwpft[nwp_model_utils.DATA_KEY].values[..., j],
                    axis=0
                )
                this_diff_matrix = numpy.concatenate(
                    [this_diff_matrix[[0], ...], this_diff_matrix], axis=0
                )
                this_hourly_diff_matrix = (
                    this_diff_matrix / nwp_forecast_hour_diffs
                )

                norm_param_dict_dict[f] = _update_norm_params_1var_1file(
                    norm_param_dict=norm_param_dict_dict[f],
                    new_data_matrix=this_hourly_diff_matrix
                )
                continue

            for k in range(len(nwp_forecast_hours)):
                if nwp_forecast_hours[k] not in precip_forecast_hours:
                    continue

                h = nwp_forecast_hours[k]

                if k == 0:
                    this_diff_matrix = (
                        nwpft[nwp_model_utils.DATA_KEY].values[k + 1, ..., j] -
                        nwpft[nwp_model_utils.DATA_KEY].values[k, ..., j]
                    )
                else:
                    this_diff_matrix = (
                        nwpft[nwp_model_utils.DATA_KEY].values[k, ..., j] -
                        nwpft[nwp_model_utils.DATA_KEY].values[k - 1, ..., j]
                    )

                this_hourly_diff_matrix = (
                    this_diff_matrix / numpy.squeeze(nwp_forecast_hour_diffs[k])
                )

                norm_param_dict_dict[f, h] = _update_norm_params_1var_1file(
                    norm_param_dict=norm_param_dict_dict[f, h],
                    new_data_matrix=this_hourly_diff_matrix
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
        nwp_model_utils.FIELD_DIM: field_names
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

    return xarray.Dataset(data_vars=main_data_dict, coords=coord_dict)


def intermediate_to_final_normalization_params(
        intermediate_norm_param_tables_xarray):
    """Computes final normalization params from intermediate ones.

    Each set of intermediate normalization params is based on a different chunk
    of data -- e.g., one possibility is that each set of intermediate params is
    based on a different NWP model.

    :param intermediate_norm_param_tables_xarray: 1-D list of xarray tables with
        intermediate normalization params, each produced by
        `get_intermediate_norm_params_for_nwp`.
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

    # Do actual stuff.
    precip_forecast_hours = (
        first_table.coords[nwp_model_utils.FORECAST_HOUR_DIM].values
    )
    field_names = first_table.coords[nwp_model_utils.FIELD_DIM].values

    num_fields = len(field_names)
    num_precip_hours = len(precip_forecast_hours)

    mean_value_matrix = numpy.full(
        (num_precip_hours, num_fields), numpy.nan
    )
    mean_squared_value_matrix = numpy.full(
        (num_precip_hours, num_fields), numpy.nan
    )
    stdev_matrix = numpy.full(
        (num_precip_hours, num_fields), numpy.nan
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

    coord_dict = {
        nwp_model_utils.FORECAST_HOUR_DIM: precip_forecast_hours,
        nwp_model_utils.FIELD_DIM: field_names
    }

    these_dim = (nwp_model_utils.FORECAST_HOUR_DIM, nwp_model_utils.FIELD_DIM)
    main_data_dict = {
        nwp_model_utils.MEAN_VALUE_KEY: (these_dim, mean_value_matrix),
        nwp_model_utils.MEAN_SQUARED_VALUE_KEY: (
            these_dim, mean_squared_value_matrix
        ),
        nwp_model_utils.STDEV_KEY: (these_dim, stdev_matrix)
    }

    return xarray.Dataset(data_vars=main_data_dict, coords=coord_dict)


def get_normalization_params_for_nwp(
        interp_nwp_file_names, non_resid_normalization_file_name, field_names,
        precip_forecast_hours):
    """Computes normalization parameters for NWP data.

    :param interp_nwp_file_names: 1-D list of paths to input files (will be read
        by `interp_nwp_model_io.read_file`).
    :param non_resid_normalization_file_name: Path to file with parameters for
        non-residual normalization.
    :param field_names: 1-D list of fields for which to compute normalization
        params.
    :param precip_forecast_hours: 1-D numpy array of forecast hours at which to
        compute normalization params for precipitation.  For all other
        variables, normalization params will be time-independent.
    :return: normalization_param_table_xarray: xarray table with normalization
        parameters.  Metadata and variable names in this table should make it
        self-explanatory.
    """

    # Check input args.
    error_checking.assert_is_string_list(interp_nwp_file_names)
    error_checking.assert_is_string_list(field_names)

    error_checking.assert_is_numpy_array(
        precip_forecast_hours, num_dimensions=1
    )
    error_checking.assert_is_integer_numpy_array(precip_forecast_hours)
    error_checking.assert_is_greater_numpy_array(precip_forecast_hours, 0)

    for this_field_name in field_names:
        nwp_model_utils.check_field_name(this_field_name)

    # Do actual stuff.
    norm_param_dict_dict = {}

    for this_field_name in field_names:
        if this_field_name not in ACCUM_PRECIP_FIELD_NAMES:
            norm_param_dict_dict[this_field_name] = {
                NUM_VALUES_KEY: 0,
                MEAN_VALUE_KEY: 0.,
                MEAN_OF_SQUARES_KEY: 0.
            }
            continue

        for this_forecast_hour in precip_forecast_hours:
            norm_param_dict_dict[this_field_name, this_forecast_hour] = {
                NUM_VALUES_KEY: 0,
                MEAN_VALUE_KEY: 0.,
                MEAN_OF_SQUARES_KEY: 0.
            }

    print((
        'Reading params for non-residual normalization from: "{0:s}"...'
    ).format(
        non_resid_normalization_file_name
    ))
    non_resid_norm_param_table_xarray = nwp_model_io.read_normalization_file(
        non_resid_normalization_file_name
    )

    already_used_file_names = []

    for i in range(len(interp_nwp_file_names)):
        if interp_nwp_file_names[i] in already_used_file_names:
            continue

        this_model_name = interp_nwp_model_io.file_name_to_model_name(
            interp_nwp_file_names[i]
        )
        this_init_time_unix_sec = interp_nwp_model_io.file_name_to_init_time(
            interp_nwp_file_names[i]
        )
        these_forecast_hours = nwp_model_utils.model_to_forecast_hours(
            model_name=this_model_name,
            init_time_unix_sec=this_init_time_unix_sec
        )
        this_directory_name = '/'.join(interp_nwp_file_names[i].split('/')[:-2])

        these_file_names = [
            interp_nwp_model_io.find_file(
                directory_name=this_directory_name,
                init_time_unix_sec=this_init_time_unix_sec,
                forecast_hour=f,
                model_name=this_model_name,
                raise_error_if_missing=False
            )
            for f in these_forecast_hours
        ]
        these_file_names = [fn for fn in these_file_names if os.path.isfile(fn)]
        already_used_file_names += these_file_names

        nwp_forecast_tables_xarray = []
        for this_file_name in these_file_names:
            print('Reading data from: "{0:s}"...'.format(this_file_name))
            nwp_forecast_tables_xarray.append(
                interp_nwp_model_io.read_file(this_file_name)
            )

        nwp_forecast_tables_xarray = [
            nwpft.drop_vars(attrs=list(nwpft.attrs.keys()))
            for nwpft in nwp_forecast_tables_xarray
        ]

        try:
            nwp_forecast_table_xarray = xarray.concat(
                nwp_forecast_tables_xarray,
                dim=nwp_model_utils.FORECAST_HOUR_DIM,
                data_vars=[nwp_model_utils.DATA_KEY],
                coords='minimal', compat='identical', join='exact'
            )
        except Exception as this_exception:
            warning_string = (
                'POTENTIAL ERROR: Could not concatenate tables:\n{0:s}'
            ).format(
                str(this_exception)
            )

            warnings.warn(warning_string)
            continue

        del nwp_forecast_tables_xarray

        nwp_forecast_table_xarray = non_resid_normalization.normalize_nwp_data(
            nwp_forecast_table_xarray=nwp_forecast_table_xarray,
            norm_param_table_xarray=non_resid_norm_param_table_xarray,
            use_quantile_norm=True
        )
        nwpft = nwp_forecast_table_xarray
        nwp_field_names = nwpft.coords[nwp_model_utils.FIELD_DIM].values
        nwp_forecast_hours = numpy.round(
            nwpft.coords[nwp_model_utils.FORECAST_HOUR_DIM].values
        ).astype(int)

        nwp_forecast_hour_diffs = numpy.diff(nwp_forecast_hours)
        nwp_forecast_hour_diffs = numpy.concatenate([
            nwp_forecast_hour_diffs[[0]], nwp_forecast_hour_diffs
        ])
        nwp_forecast_hour_diffs = numpy.expand_dims(
            nwp_forecast_hour_diffs, axis=-1
        )
        nwp_forecast_hour_diffs = numpy.expand_dims(
            nwp_forecast_hour_diffs, axis=-1
        )
        nwp_forecast_hour_diffs[nwp_forecast_hour_diffs > 12] = numpy.nan

        for j in range(len(nwp_field_names)):
            f = nwp_field_names[j]
            if f not in field_names:
                continue

            if f not in ACCUM_PRECIP_FIELD_NAMES:
                this_diff_matrix = numpy.diff(
                    nwpft[nwp_model_utils.DATA_KEY].values[..., j],
                    axis=0
                )
                this_diff_matrix = numpy.concatenate(
                    [this_diff_matrix[[0], ...], this_diff_matrix], axis=0
                )
                this_hourly_diff_matrix = (
                    this_diff_matrix / nwp_forecast_hour_diffs
                )

                norm_param_dict_dict[f] = _update_norm_params_1var_1file(
                    norm_param_dict=norm_param_dict_dict[f],
                    new_data_matrix=this_hourly_diff_matrix
                )
                continue

            for k in range(len(nwp_forecast_hours)):
                if nwp_forecast_hours[k] not in precip_forecast_hours:
                    continue

                h = nwp_forecast_hours[k]

                if k == 0:
                    this_diff_matrix = (
                        nwpft[nwp_model_utils.DATA_KEY].values[k + 1, ..., j] -
                        nwpft[nwp_model_utils.DATA_KEY].values[k, ..., j]
                    )
                else:
                    this_diff_matrix = (
                        nwpft[nwp_model_utils.DATA_KEY].values[k, ..., j] -
                        nwpft[nwp_model_utils.DATA_KEY].values[k - 1, ..., j]
                    )

                this_hourly_diff_matrix = (
                    this_diff_matrix / numpy.squeeze(nwp_forecast_hour_diffs[k])
                )

                norm_param_dict_dict[f, h] = _update_norm_params_1var_1file(
                    norm_param_dict=norm_param_dict_dict[f, h],
                    new_data_matrix=this_hourly_diff_matrix
                )

    num_fields = len(field_names)
    num_precip_hours = len(precip_forecast_hours)

    mean_value_matrix = numpy.full(
        (num_precip_hours, num_fields), numpy.nan
    )
    mean_squared_value_matrix = numpy.full(
        (num_precip_hours, num_fields), numpy.nan
    )
    stdev_matrix = numpy.full(
        (num_precip_hours, num_fields), numpy.nan
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

    coord_dict = {
        nwp_model_utils.FORECAST_HOUR_DIM: precip_forecast_hours,
        nwp_model_utils.FIELD_DIM: field_names
    }

    these_dim = (nwp_model_utils.FORECAST_HOUR_DIM, nwp_model_utils.FIELD_DIM)
    main_data_dict = {
        nwp_model_utils.MEAN_VALUE_KEY: (these_dim, mean_value_matrix),
        nwp_model_utils.MEAN_SQUARED_VALUE_KEY: (
            these_dim, mean_squared_value_matrix
        ),
        nwp_model_utils.STDEV_KEY: (these_dim, stdev_matrix)
    }

    return xarray.Dataset(data_vars=main_data_dict, coords=coord_dict)


def normalize_nwp_data(nwp_forecast_table_xarray, norm_param_table_xarray):
    """Normalizes NWP data.

    :param nwp_forecast_table_xarray: xarray table with NWP data in z-score
        units, after non-residual normalization.
    :param norm_param_table_xarray: xarray table with normalization
        parameters, created by `get_normalization_params_for_nwp`.
    :return: nwp_forecast_table_xarray: Same as input but in residual scores.
    """

    nwpft = nwp_forecast_table_xarray
    npt = norm_param_table_xarray

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
            data_matrix[..., j] = _normalize_1var(
                z_score_values=data_matrix[..., j],
                reference_temporal_stdev=
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

            data_matrix[k, ..., j] = _normalize_1var(
                z_score_values=data_matrix[k, ..., j],
                reference_temporal_stdev=
                npt[nwp_model_utils.STDEV_KEY].values[k_new, j_new]
            )

    return nwp_forecast_table_xarray.assign({
        nwp_model_utils.DATA_KEY: (
            nwp_forecast_table_xarray[nwp_model_utils.DATA_KEY].dims,
            data_matrix
        )
    })


def denormalize_nwp_data(nwp_forecast_table_xarray, norm_param_table_xarray):
    """Denormalizes NWP data.

    :param nwp_forecast_table_xarray: xarray table with NWP data in residual
        scores.
    :param norm_param_table_xarray: xarray table with normalization
        parameters, created by `get_normalization_params_for_nwp`.
    :return: nwp_forecast_table_xarray: Same as input but in z-scores.
    """

    nwpft = nwp_forecast_table_xarray
    npt = norm_param_table_xarray

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
            data_matrix[..., j] = _denormalize_1var(
                residual_values=data_matrix[..., j],
                reference_temporal_stdev=
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

            data_matrix[k, ..., j] = _denormalize_1var(
                residual_values=data_matrix[k, ..., j],
                reference_temporal_stdev=
                npt[nwp_model_utils.STDEV_KEY].values[k_new, j_new]
            )

    return nwp_forecast_table_xarray.assign({
        nwp_model_utils.DATA_KEY: (
            nwp_forecast_table_xarray[nwp_model_utils.DATA_KEY].dims,
            data_matrix
        )
    })


def get_normalization_params_for_targets(urma_file_names,
                                         non_resid_normalization_file_name):
    """Computes normalization params for each URMA target variable.

    :param urma_file_names: 1-D list of paths to URMA files (will be read by
        `urma_io.read_file`).
    :param non_resid_normalization_file_name: Path to file with parameters for
        non-residual normalization.
    :return: normalization_param_table_xarray: xarray table with normalization
        parameters.  Metadata and variable names in this table should make it
        self-explanatory.
    """

    # Check input args.
    error_checking.assert_is_string_list(urma_file_names)

    # Housekeeping.
    first_urma_table_xarray = urma_io.read_file(urma_file_names[0])
    field_names = (
        first_urma_table_xarray.coords[urma_utils.FIELD_DIM].values.tolist()
    )

    norm_param_dict_dict = {}
    for this_field_name in field_names:
        norm_param_dict_dict[this_field_name] = {
            NUM_VALUES_KEY: 0,
            MEAN_VALUE_KEY: 0.,
            MEAN_OF_SQUARES_KEY: 0.
        }

    # Do actual stuff.
    print((
        'Reading parameters for non-residual normalization from: "{0:s}"...'
    ).format(
        non_resid_normalization_file_name
    ))
    non_resid_norm_param_table_xarray = urma_io.read_normalization_file(
        non_resid_normalization_file_name
    )

    for i in range(len(urma_file_names)):
        print('Reading data from: "{0:s}"...'.format(urma_file_names[i]))
        this_urma_table_xarray = urma_io.read_file(urma_file_names[i])
        this_urma_table_xarray = non_resid_normalization.normalize_targets(
            urma_table_xarray=this_urma_table_xarray,
            norm_param_table_xarray=non_resid_norm_param_table_xarray,
            use_quantile_norm=True
        )
        tutx = this_urma_table_xarray

        valid_hours = (
            SECONDS_TO_HOURS * tutx.coords[urma_utils.VALID_TIME_DIM].values
        )
        valid_hour_diffs = numpy.diff(valid_hours)
        valid_hour_diffs = numpy.concatenate([
            valid_hour_diffs[[0]], valid_hour_diffs
        ])
        valid_hour_diffs = numpy.expand_dims(valid_hour_diffs, axis=-1)
        valid_hour_diffs = numpy.expand_dims(valid_hour_diffs, axis=-1)

        for j in range(len(tutx.coords[urma_utils.FIELD_DIM].values)):
            f = tutx.coords[urma_utils.FIELD_DIM].values[j]

            this_diff_matrix = numpy.diff(
                tutx[urma_utils.DATA_KEY].values[..., j],
                axis=0
            )
            this_diff_matrix = numpy.concatenate(
                [this_diff_matrix[[0], ...], this_diff_matrix], axis=0
            )
            this_hourly_diff_matrix = (
                this_diff_matrix / valid_hour_diffs
            )

            norm_param_dict_dict[f] = _update_norm_params_1var_1file(
                norm_param_dict=norm_param_dict_dict[f],
                new_data_matrix=this_hourly_diff_matrix
            )

    num_fields = len(field_names)
    mean_values = numpy.full(num_fields, numpy.nan)
    mean_squared_values = numpy.full(num_fields, numpy.nan)
    stdev_values = numpy.full(num_fields, numpy.nan)

    for j in range(num_fields):
        f = field_names[j]

        mean_values[j] = norm_param_dict_dict[f][MEAN_VALUE_KEY]
        mean_squared_values[j] = norm_param_dict_dict[f][MEAN_OF_SQUARES_KEY]
        stdev_values[j] = _get_standard_deviation_1var(norm_param_dict_dict[f])

        print((
            'Mean, mean square, and standard deviation for {0:s} = '
            '{1:.4g}, {2:.4g}, {3:.4g}'
        ).format(
            field_names[j],
            mean_values[j], mean_squared_values[j], stdev_values[j]
        ))

    coord_dict = {
        urma_utils.FIELD_DIM: field_names
    }

    these_dim = (urma_utils.FIELD_DIM,)
    main_data_dict = {
        urma_utils.MEAN_VALUE_KEY: (these_dim, mean_values),
        urma_utils.MEAN_SQUARED_VALUE_KEY: (these_dim, mean_squared_values),
        urma_utils.STDEV_KEY: (these_dim, stdev_values)
    }

    return xarray.Dataset(data_vars=main_data_dict, coords=coord_dict)


def normalize_targets(urma_table_xarray, norm_param_table_xarray):
    """Normalizes target variables from z-scores to residual scores.

    :param urma_table_xarray: xarray table with URMA data in z-scores.
    :param norm_param_table_xarray: xarray table with normalization
        parameters, created by `get_normalization_params_for_targets`.
    :return: urma_table_xarray: Same as input but in residual scores.
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

        data_matrix[..., j] = _normalize_1var(
            z_score_values=data_matrix[..., j],
            reference_temporal_stdev=npt[urma_utils.STDEV_KEY].values[j_new]
        )

    return urma_table_xarray.assign({
        urma_utils.DATA_KEY: (
            urma_table_xarray[urma_utils.DATA_KEY].dims, data_matrix
        )
    })


def denormalize_targets(urma_table_xarray, norm_param_table_xarray):
    """Denormalizes target variables from residual scores to z-scores.

    :param urma_table_xarray: xarray table with URMA data in residual scores.
    :param norm_param_table_xarray: xarray table with normalization
        parameters, created by `get_normalization_params_for_targets`.
    :return: urma_table_xarray: Same as input but in z-scores.
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

        data_matrix[..., j] = _denormalize_1var(
            residual_values=data_matrix[..., j],
            reference_temporal_stdev=npt[urma_utils.STDEV_KEY].values[j_new]
        )

    return urma_table_xarray.assign({
        urma_utils.DATA_KEY: (
            urma_table_xarray[urma_utils.DATA_KEY].dims, data_matrix
        )
    })

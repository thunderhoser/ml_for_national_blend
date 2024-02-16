"""Helper methods for normalization."""

import numpy
import xarray
from gewittergefahr.gg_utils import error_checking
from ml_for_national_blend.io import urma_io
from ml_for_national_blend.io import interp_nwp_model_io
from ml_for_national_blend.utils import urma_utils
from ml_for_national_blend.utils import nwp_model_utils

# TODO(thunderhoser): I would still like to implement quantile normalization.

NUM_VALUES_KEY = 'num_values'
MEAN_VALUE_KEY = 'mean_value'
MEAN_OF_SQUARES_KEY = 'mean_of_squares'

ACCUM_PRECIP_FIELD_NAMES = [nwp_model_utils.PRECIP_NAME]


def _update_z_score_params(z_score_param_dict, new_data_matrix):
    """Updates z-score parameters.

    :param z_score_param_dict: Dictionary with the following keys.
    z_score_param_dict['num_values']: Number of values on which current
        estimates are based.
    z_score_param_dict['mean_value']: Current mean.
    z_score_param_dict['mean_of_squares']: Current mean of squared values.

    :param new_data_matrix: numpy array with new values.  Will be used to
        update estimates in `z_score_param_dict`.
    :return: z_score_param_dict: Same as input, but with new estimates.
    """

    if numpy.all(numpy.isnan(new_data_matrix)):
        return z_score_param_dict

    new_num_values = numpy.sum(numpy.invert(numpy.isnan(new_data_matrix)))

    these_means = numpy.array([
        z_score_param_dict[MEAN_VALUE_KEY],
        numpy.nanmean(new_data_matrix)
    ])
    these_weights = numpy.array([
        z_score_param_dict[NUM_VALUES_KEY], new_num_values
    ])
    z_score_param_dict[MEAN_VALUE_KEY] = numpy.average(
        these_means, weights=these_weights)

    these_means = numpy.array([
        z_score_param_dict[MEAN_OF_SQUARES_KEY],
        numpy.nanmean(new_data_matrix ** 2)
    ])
    these_weights = numpy.array([
        z_score_param_dict[NUM_VALUES_KEY], new_num_values
    ])
    z_score_param_dict[MEAN_OF_SQUARES_KEY] = numpy.average(
        these_means, weights=these_weights)

    z_score_param_dict[NUM_VALUES_KEY] += new_num_values
    return z_score_param_dict


def _get_standard_deviation(z_score_param_dict):
    """Computes standard deviation.

    :param z_score_param_dict: See doc for `_update_z_score_params`.
    :return: standard_deviation: Standard deviation.
    """

    if z_score_param_dict[NUM_VALUES_KEY] == 0:
        return numpy.nan

    multiplier = float(
        z_score_param_dict[NUM_VALUES_KEY]
    ) / (z_score_param_dict[NUM_VALUES_KEY] - 1)

    return numpy.sqrt(multiplier * (
        z_score_param_dict[MEAN_OF_SQUARES_KEY] -
        z_score_param_dict[MEAN_VALUE_KEY] ** 2
    ))


def get_z_score_params_for_nwp(interp_nwp_file_names, field_names,
                               precip_forecast_hours):
    """Computes z-score parameters for NWP data.

    :param interp_nwp_file_names: 1-D list of paths to input files (will be read
        by `interp_nwp_model_io.read_file`).
    :param field_names: 1-D list of fields for which to compute z-score
        parameters.
    :param precip_forecast_hours: 1-D numpy array of forecast hours at which to
        compute z-score parameters for precipitation.  For all other variables,
        z-score parameters will be time-independent.
    :return: z_score_param_table_xarray: xarray table with z-score parameters.
        Metadata and variable names in this table should make it self-
        explanatory.
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
    z_score_dict_dict = {}

    for this_field_name in field_names:
        if this_field_name not in ACCUM_PRECIP_FIELD_NAMES:
            z_score_dict_dict[this_field_name] = {
                NUM_VALUES_KEY: 0,
                MEAN_VALUE_KEY: 0.,
                MEAN_OF_SQUARES_KEY: 0.
            }
            continue

        for this_forecast_hour in precip_forecast_hours:
            z_score_dict_dict[this_field_name, this_forecast_hour] = {
                NUM_VALUES_KEY: 0,
                MEAN_VALUE_KEY: 0.,
                MEAN_OF_SQUARES_KEY: 0.
            }

    for this_file_name in interp_nwp_file_names:
        print('Reading data from: "{0:s}"...'.format(this_file_name))
        nwp_forecast_table_xarray = interp_nwp_model_io.read_file(
            this_file_name
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
                z_score_dict_dict[f] = _update_z_score_params(
                    z_score_param_dict=z_score_dict_dict[f],
                    new_data_matrix=
                    nwpft[nwp_model_utils.DATA_KEY].values[..., j]
                )
                continue

            for k in range(len(nwp_forecast_hours)):
                if nwp_forecast_hours[k] not in precip_forecast_hours:
                    continue

                h = nwp_forecast_hours[k]

                z_score_dict_dict[f, h] = _update_z_score_params(
                    z_score_param_dict=z_score_dict_dict[f, h],
                    new_data_matrix=
                    nwpft[nwp_model_utils.DATA_KEY].values[k, ..., j]
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
                    z_score_dict_dict[f, h][MEAN_VALUE_KEY]
                )
                mean_squared_value_matrix[k, j] = (
                    z_score_dict_dict[f, h][MEAN_OF_SQUARES_KEY]
                )
                stdev_matrix[k, j] = _get_standard_deviation(
                    z_score_dict_dict[f, h]
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
                mean_value_matrix[k, j] = z_score_dict_dict[f][MEAN_VALUE_KEY]
                mean_squared_value_matrix[k, j] = (
                    z_score_dict_dict[f][MEAN_OF_SQUARES_KEY]
                )
                stdev_matrix[k, j] = _get_standard_deviation(
                    z_score_dict_dict[f]
                )

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


def normalize_nwp_data_to_z_scores(nwp_forecast_table_xarray,
                                   z_score_param_table_xarray):
    """Normalizes NWP data from physical units to z-scores.

    :param nwp_forecast_table_xarray: xarray table with NWP output in physical
        units.
    :param z_score_param_table_xarray: xarray table with normalization
        parameters (means and standard deviations), created by
        `get_z_score_params_for_nwp`.
    :return: nwp_forecast_table_xarray: Same as input but in z-score units.
    """

    # TODO(thunderhoser): Still need unit test.

    nwpft = nwp_forecast_table_xarray
    zspt = z_score_param_table_xarray

    field_names = nwpft.coords[nwp_model_utils.FIELD_DIM].values.tolist()
    forecast_hours = numpy.round(
        nwpft.coords[nwp_model_utils.FORECAST_HOUR_DIM].values
    ).astype(int)

    num_forecast_hours = len(forecast_hours)
    num_fields = len(field_names)
    data_matrix = nwpft[nwp_model_utils.DATA_KEY].values

    for j in range(num_fields):
        j_new = numpy.where(
            zspt.coords[nwp_model_utils.FIELD_DIM].values == field_names[j]
        )[0][0]

        if field_names[j] not in ACCUM_PRECIP_FIELD_NAMES:
            this_mean = zspt[nwp_model_utils.MEAN_VALUE_KEY].values[0, j_new]
            this_stdev = zspt[nwp_model_utils.STDEV_KEY].values[0, j_new]

            if numpy.isnan(this_stdev):
                data_matrix[..., j] = 0.
            else:
                data_matrix[..., j] = (
                    (data_matrix[..., j] - this_mean) / this_stdev
                )

            continue

        for k in range(num_forecast_hours):
            k_new = numpy.where(
                numpy.round(
                    zspt.coords[nwp_model_utils.FORECAST_HOUR_DIM].values
                ).astype(int)
                == forecast_hours[k]
            )[0][0]

            this_mean = zspt[nwp_model_utils.MEAN_VALUE_KEY].values[
                k_new, j_new
            ]
            this_stdev = zspt[nwp_model_utils.STDEV_KEY].values[k_new, j_new]

            if numpy.isnan(this_stdev):
                data_matrix[k, ..., j] = 0.
            else:
                data_matrix[k, ..., j] = (
                    (data_matrix[k, ..., j] - this_mean) / this_stdev
                )

    try:
        nwp_forecast_table_xarray = nwp_forecast_table_xarray.assign({
            nwp_model_utils.DATA_KEY: (
                nwp_forecast_table_xarray[nwp_model_utils.DATA_KEY].dims,
                data_matrix
            )
        })
    except:
        data_key = nwp_model_utils.DATA_KEY

        nwp_forecast_table_xarray = nwp_forecast_table_xarray.assign(
            data_key=(
                nwp_forecast_table_xarray[data_key].dims, data_matrix
            )
        )

    return nwp_forecast_table_xarray


def denormalize_nwp_data_from_z_scores(nwp_forecast_table_xarray,
                                       z_score_param_table_xarray):
    """Denormalizes NWP data from z-scores to physical units.

    :param nwp_forecast_table_xarray: xarray table with NWP output in z-scores.
    :param z_score_param_table_xarray: xarray table with normalization
        parameters (means and standard deviations), created by
        `get_z_score_params_for_nwp`.
    :return: nwp_forecast_table_xarray: Same as input but in physical units.
    """

    # TODO(thunderhoser): Still need unit test.

    nwpft = nwp_forecast_table_xarray
    zspt = z_score_param_table_xarray

    field_names = nwpft.coords[nwp_model_utils.FIELD_DIM].values.tolist()
    forecast_hours = numpy.round(
        nwpft.coords[nwp_model_utils.FORECAST_HOUR_DIM].values
    ).astype(int)

    num_forecast_hours = len(forecast_hours)
    num_fields = len(field_names)
    data_matrix = nwpft[nwp_model_utils.DATA_KEY].values

    for j in range(num_fields):
        j_new = numpy.where(
            zspt.coords[nwp_model_utils.FIELD_DIM].values == field_names[j]
        )[0][0]

        if field_names[j] not in ACCUM_PRECIP_FIELD_NAMES:
            this_mean = zspt[nwp_model_utils.MEAN_VALUE_KEY].values[0, j_new]
            this_stdev = zspt[nwp_model_utils.STDEV_KEY].values[0, j_new]

            if numpy.isnan(this_stdev):
                data_matrix[..., j] = this_mean
            else:
                data_matrix[..., j] = (
                    this_mean + this_stdev * data_matrix[..., j]
                )

            continue

        for k in range(num_forecast_hours):
            k_new = numpy.where(
                numpy.round(
                    zspt.coords[nwp_model_utils.FORECAST_HOUR_DIM].values
                ).astype(int)
                == forecast_hours[k]
            )[0][0]

            this_mean = zspt[nwp_model_utils.MEAN_VALUE_KEY].values[
                k_new, j_new
            ]
            this_stdev = zspt[nwp_model_utils.STDEV_KEY].values[k_new, j_new]

            if numpy.isnan(this_stdev):
                data_matrix[k, ..., j] = this_mean
            else:
                data_matrix[k, ..., j] = (
                    this_mean + this_stdev * data_matrix[k, ..., j]
                )

    try:
        nwp_forecast_table_xarray = nwp_forecast_table_xarray.assign({
            nwp_model_utils.DATA_KEY: (
                nwp_forecast_table_xarray[nwp_model_utils.DATA_KEY].dims,
                data_matrix
            )
        })
    except:
        data_key = nwp_model_utils.DATA_KEY

        nwp_forecast_table_xarray = nwp_forecast_table_xarray.assign(
            data_key=(
                nwp_forecast_table_xarray[data_key].dims, data_matrix
            )
        )

    return nwp_forecast_table_xarray


def get_z_score_params_for_targets(urma_file_names):
    """Computes z-score parameters for each URMA target variable.

    :param urma_file_names: 1-D list of paths to URMA files (will be read by
        `urma_io.read_file`).
    :return: z_score_param_table_xarray: xarray table with z-score parameters.
        Metadata and variable names in this table should make it self-
        explanatory.
    """

    first_urma_table_xarray = urma_io.read_file(urma_file_names[0])
    field_names = (
        first_urma_table_xarray.coords[urma_utils.FIELD_DIM].values.tolist()
    )

    z_score_dict_dict = {}
    for this_field_name in field_names:
        z_score_dict_dict[this_field_name] = {
            NUM_VALUES_KEY: 0,
            MEAN_VALUE_KEY: 0.,
            MEAN_OF_SQUARES_KEY: 0.
        }

    for this_file_name in urma_file_names:
        print('Reading data from: "{0:s}"...'.format(this_file_name))
        this_urma_table_xarray = urma_io.read_file(this_file_name)
        tutx = this_urma_table_xarray

        for j in range(len(tutx.coords[urma_utils.FIELD_DIM].values)):
            f = tutx.coords[urma_utils.FIELD_DIM].values[j]

            z_score_dict_dict[f] = _update_z_score_params(
                z_score_param_dict=z_score_dict_dict[f],
                new_data_matrix=tutx[urma_utils.DATA_KEY].values[..., j]
            )

    num_fields = len(field_names)
    mean_values = numpy.full(num_fields, numpy.nan)
    mean_squared_values = numpy.full(num_fields, numpy.nan)
    stdev_values = numpy.full(num_fields, numpy.nan)

    for j in range(num_fields):
        f = field_names[j]

        mean_values[j] = z_score_dict_dict[f][MEAN_VALUE_KEY]
        mean_squared_values[j] = z_score_dict_dict[f][MEAN_OF_SQUARES_KEY]
        stdev_values[j] = _get_standard_deviation(z_score_dict_dict[f])

        print((
            'Mean, mean square, and standard deviation for {0:s} = '
            '{1:.4g}, {2:.4g}, {3:.4g}'
        ).format(
            field_names[j],
            mean_values[j], mean_squared_values[j], stdev_values[j]
        ))

    coord_dict = {urma_utils.FIELD_DIM: field_names}

    these_dim = (urma_utils.FIELD_DIM,)
    main_data_dict = {
        urma_utils.MEAN_VALUE_KEY: (these_dim, mean_values),
        urma_utils.MEAN_SQUARED_VALUE_KEY: (these_dim, mean_squared_values),
        urma_utils.STDEV_KEY: (these_dim, stdev_values)
    }

    return xarray.Dataset(data_vars=main_data_dict, coords=coord_dict)


def normalize_targets_to_z_scores(urma_table_xarray,
                                  z_score_param_table_xarray):
    """Normalizes target variables from physical units to z-scores.

    :param urma_table_xarray: xarray table with URMA data in physical units.
    :param z_score_param_table_xarray: xarray table with normalization
        parameters (means and standard deviations), created by
        `get_z_score_params_for_targets`.
    :return: urma_table_xarray: Same as input but in z-score units.
    """

    # TODO(thunderhoser): Still need unit test.

    urmat = urma_table_xarray
    zspt = z_score_param_table_xarray

    field_names = urmat.coords[urma_utils.FIELD_DIM].values.tolist()
    num_fields = len(field_names)

    data_matrix = urmat[urma_utils.DATA_KEY].values

    for j in range(num_fields):
        j_new = numpy.where(
            zspt.coords[urma_utils.FIELD_DIM].values == field_names[j]
        )[0][0]

        this_mean = zspt[urma_utils.MEAN_VALUE_KEY].values[j_new]
        this_stdev = zspt[urma_utils.STDEV_KEY].values[j_new]

        if numpy.isnan(this_stdev):
            data_matrix[..., j] = 0.
        else:
            data_matrix[..., j] = (data_matrix[..., j] - this_mean) / this_stdev

    try:
        urma_table_xarray = urma_table_xarray.assign({
            urma_utils.DATA_KEY: (
                urma_table_xarray[urma_utils.DATA_KEY].dims, data_matrix
            )
        })
    except:
        data_key = urma_utils.DATA_KEY

        urma_table_xarray = urma_table_xarray.assign(
            data_key=(urma_table_xarray[data_key].dims, data_matrix)
        )

    return urma_table_xarray


def denormalize_targets_to_z_scores(urma_table_xarray,
                                    z_score_param_table_xarray):
    """Denormalizes target variables from z-scores to physical units.

    :param urma_table_xarray: xarray table with URMA data in z-scores.
    :param z_score_param_table_xarray: xarray table with normalization
        parameters (means and standard deviations), created by
        `get_z_score_params_for_targets`.
    :return: urma_table_xarray: Same as input but in physical units.
    """

    # TODO(thunderhoser): Still need unit test.

    urmat = urma_table_xarray
    zspt = z_score_param_table_xarray

    field_names = urmat.coords[urma_utils.FIELD_DIM].values.tolist()
    num_fields = len(field_names)

    data_matrix = urmat[urma_utils.DATA_KEY].values

    for j in range(num_fields):
        j_new = numpy.where(
            zspt.coords[urma_utils.FIELD_DIM].values == field_names[j]
        )[0][0]

        this_mean = zspt[urma_utils.MEAN_VALUE_KEY].values[j_new]
        this_stdev = zspt[urma_utils.STDEV_KEY].values[j_new]

        if numpy.isnan(this_stdev):
            data_matrix[..., j] = this_mean
        else:
            data_matrix[..., j] = this_mean + this_stdev * data_matrix[..., j]

    try:
        urma_table_xarray = urma_table_xarray.assign({
            urma_utils.DATA_KEY: (
                urma_table_xarray[urma_utils.DATA_KEY].dims, data_matrix
            )
        })
    except:
        data_key = urma_utils.DATA_KEY

        urma_table_xarray = urma_table_xarray.assign(
            data_key=(urma_table_xarray[data_key].dims, data_matrix)
        )

    return urma_table_xarray

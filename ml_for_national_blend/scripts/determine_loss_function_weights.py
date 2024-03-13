"""Determines weights for loss function.

There is one weight per target variable; the goal is to make every target
variable have an equal contribution to the overall loss.  This script assumes
that the loss function is MSE or dual-weighted MSE.
"""

import argparse
import numpy
from gewittergefahr.gg_utils import time_conversion
from gewittergefahr.gg_utils import temperature_conversions as temperature_conv
from ml_for_national_blend.io import urma_io
from ml_for_national_blend.utils import urma_utils

INPUT_DIR_ARG_NAME = 'input_target_dir_name'
TARGET_FIELDS_ARG_NAME = 'target_field_names'
FIRST_DATES_ARG_NAME = 'first_valid_date_strings'
LAST_DATES_ARG_NAME = 'last_valid_date_strings'
NORMALIZATION_FILE_ARG_NAME = 'input_normalization_file_name'
USE_DWMSE_ARG_NAME = 'use_dwmse'
NUM_SAMPLE_VALUES_ARG_NAME = 'num_sample_values_per_file'

INPUT_DIR_HELP_STRING = (
    'Path to directory with target fields from URMA.  Files therein will be '
    'found by `urma_io.find_file` and read by `urma_io.read_file`.'
)
TARGET_FIELDS_HELP_STRING = '1-D list with names of target fields to predict.'
FIRST_DATES_HELP_STRING = (
    'List with first valid date (format "yyyymmdd") for each continuous '
    'period.  Weights will be based on all valid dates in all periods.'
)
LAST_DATES_HELP_STRING = 'See documentation for {0:s}.'.format(
    FIRST_DATES_ARG_NAME
)
NORMALIZATION_FILE_HELP_STRING = (
    'Path to normalization file.  For each target field, the mean value -- '
    'used to create a climo model -- will be read from here.'
)
USE_DWMSE_HELP_STRING = (
    'Boolean flag.  If 1, will use dual-weighted MSE.  If 0, will use vanilla '
    'MSE.'
)
NUM_SAMPLE_VALUES_HELP_STRING = (
    'Number of sample values per file to use for computing weights.  This '
    'value will be applied to each variable.'
)

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + INPUT_DIR_ARG_NAME, type=str, required=True,
    help=INPUT_DIR_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + TARGET_FIELDS_ARG_NAME, type=str, nargs='+', required=True,
    help=TARGET_FIELDS_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + FIRST_DATES_ARG_NAME, type=str, nargs='+', required=True,
    help=FIRST_DATES_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + LAST_DATES_ARG_NAME, type=str, nargs='+', required=True,
    help=LAST_DATES_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + NORMALIZATION_FILE_ARG_NAME, type=str, required=True,
    help=NORMALIZATION_FILE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + USE_DWMSE_ARG_NAME, type=int, required=True,
    help=USE_DWMSE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + NUM_SAMPLE_VALUES_ARG_NAME, type=int, required=True,
    help=NUM_SAMPLE_VALUES_HELP_STRING
)


def _increment_dwmse_one_field(urma_table_xarray, field_name,
                               climo_mean, num_sample_values):
    """Increments DWMSE (dual-weighted mean squared error) for one field.

    :param urma_table_xarray: xarray table in format returned by
        `urma_io.read_file`.
    :param field_name: Name of field.
    :param climo_mean: Climatological mean for given field.
    :param num_sample_values: Number of sample values to use from
        `urma_table_xarray`.
    :return: new_dwmse: DWMSE for new data batch.
    :return: new_num_values: Number of values in new data batch.
    """

    k = numpy.where(
        urma_table_xarray.coords[urma_utils.FIELD_DIM].values == field_name
    )[0][0]
    data_matrix = urma_table_xarray[urma_utils.DATA_KEY].values[..., k]

    real_data_values = data_matrix[numpy.invert(numpy.isnan(data_matrix))]
    numpy.random.shuffle(real_data_values)
    real_data_values = real_data_values[:num_sample_values]

    sample_weights = numpy.maximum(
        numpy.absolute(climo_mean),
        numpy.absolute(real_data_values)
    )

    if field_name in [
            urma_utils.TEMPERATURE_2METRE_NAME, urma_utils.DEWPOINT_2METRE_NAME
    ]:
        sample_weights = temperature_conv.kelvins_to_celsius(sample_weights)

    # sample_weights = numpy.minimum(sample_weights, extreme_threshold)
    error_values = sample_weights * (climo_mean - real_data_values) ** 2

    return numpy.mean(error_values), len(error_values)


def _increment_mse_one_field(urma_table_xarray, field_name, climo_mean,
                             num_sample_values):
    """Increments MSE for one field.

    :param urma_table_xarray: xarray table in format returned by
        `urma_io.read_file`.
    :param field_name: Name of field.
    :param climo_mean: Climatological mean for given field.
    :param num_sample_values: Number of sample values to use from
        `urma_table_xarray`.
    :return: new_mse: MSE for new data batch.
    :return: new_num_values: Number of values in new data batch.
    """

    k = numpy.where(
        urma_table_xarray.coords[urma_utils.FIELD_DIM].values == field_name
    )[0][0]
    data_matrix = urma_table_xarray[urma_utils.DATA_KEY].values[..., k]

    real_data_values = data_matrix[numpy.invert(numpy.isnan(data_matrix))]
    numpy.random.shuffle(real_data_values)
    real_data_values = real_data_values[:num_sample_values]

    error_values = (climo_mean - real_data_values) ** 2

    return numpy.mean(error_values), len(error_values)


def _run(input_dir_name, target_field_names,
         first_date_strings, last_date_strings,
         normalization_file_name, use_dwmse, num_sample_values_per_file):
    """Determines weights for loss function.

    This is effectively the main method.

    :param input_dir_name: See documentation at top of this script.
    :param target_field_names: Same.
    :param first_date_strings: Same.
    :param last_date_strings: Same.
    :param normalization_file_name: Same.
    :param use_dwmse: Same.
    :param num_sample_values_per_file: Same.
    """

    # Check input args.
    num_periods = len(first_date_strings)
    assert len(last_date_strings) == num_periods

    # Determine the mean for each climo model.
    print('Reading data from: "{0:s}"...'.format(normalization_file_name))
    norm_param_table_xarray = urma_io.read_normalization_file(
        normalization_file_name
    )
    npt = norm_param_table_xarray

    num_fields = len(target_field_names)
    climo_mean_by_field = numpy.full(num_fields, numpy.nan)

    for j in range(num_fields):
        j_new = numpy.where(
            npt.coords[urma_utils.FIELD_DIM].values == target_field_names[j]
        )[0][0]
        climo_mean_by_field[j] = npt[urma_utils.MEAN_VALUE_KEY].values[j_new]

    # Find the weights.
    valid_date_strings = []
    for i in range(num_periods):
        valid_date_strings += time_conversion.get_spc_dates_in_range(
            first_date_strings[i], last_date_strings[i]
        )

    target_file_names = [
        urma_io.find_file(
            directory_name=input_dir_name, valid_date_string=d,
            raise_error_if_missing=True
        ) for d in valid_date_strings
    ]

    num_files = len(target_file_names)
    mean_loss_by_field = numpy.full(num_fields, numpy.nan)
    num_values_by_field = numpy.full(num_fields, 0, dtype=int)

    for i in range(num_files):
        print('Reading data from: "{0:s}"...'.format(target_file_names[i]))
        urma_table_xarray = urma_io.read_file(target_file_names[i])

        for j in range(num_fields):
            if use_dwmse:
                new_loss, new_num_values = _increment_dwmse_one_field(
                    urma_table_xarray=urma_table_xarray,
                    field_name=target_field_names[j],
                    climo_mean=climo_mean_by_field[j],
                    num_sample_values=num_sample_values_per_file
                )
            else:
                new_loss, new_num_values = _increment_mse_one_field(
                    urma_table_xarray=urma_table_xarray,
                    field_name=target_field_names[j],
                    climo_mean=climo_mean_by_field[j],
                    num_sample_values=num_sample_values_per_file
                )

            if num_values_by_field[j] == 0:
                mean_loss_by_field[j] = new_loss + 0.
            else:
                these_means = numpy.array([mean_loss_by_field[j], new_loss])
                these_weights = numpy.array([
                    num_values_by_field[j], new_num_values
                ])
                mean_loss_by_field[j] = numpy.average(
                    these_means, weights=these_weights
                )

            num_values_by_field[j] += new_num_values

    unnorm_weight_by_field = numpy.sum(mean_loss_by_field) / mean_loss_by_field
    weight_by_field = unnorm_weight_by_field / numpy.sum(unnorm_weight_by_field)

    for j in range(num_fields):
        print((
            '{0:s} ... {1:s} = {2:.2f} ... climo mean = {3:.4f} ... '
            'unnormalized weight = {4:.4f} ... normalized weight = {5:.8f}'
        ).format(
            target_field_names[j],
            'DWMSE' if use_dwmse else 'MSE',
            mean_loss_by_field[j],
            climo_mean_by_field[j],
            unnorm_weight_by_field[j],
            weight_by_field[j]
        ))


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        input_dir_name=getattr(INPUT_ARG_OBJECT, INPUT_DIR_ARG_NAME),
        target_field_names=getattr(INPUT_ARG_OBJECT, TARGET_FIELDS_ARG_NAME),
        first_date_strings=getattr(INPUT_ARG_OBJECT, FIRST_DATES_ARG_NAME),
        last_date_strings=getattr(INPUT_ARG_OBJECT, LAST_DATES_ARG_NAME),
        normalization_file_name=getattr(
            INPUT_ARG_OBJECT, NORMALIZATION_FILE_ARG_NAME
        ),
        use_dwmse=bool(getattr(INPUT_ARG_OBJECT, USE_DWMSE_ARG_NAME)),
        num_sample_values_per_file=getattr(
            INPUT_ARG_OBJECT, NUM_SAMPLE_VALUES_ARG_NAME
        )
    )

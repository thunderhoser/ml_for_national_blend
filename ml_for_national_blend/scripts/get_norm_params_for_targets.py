"""Computes normalization parameters for URMA target variables.

Normalization parameters = mean, stdev, and quantiles for each variable.
"""

import argparse
import numpy
from ml_for_national_blend.io import urma_io
from ml_for_national_blend.utils import normalization

DATE_FORMAT = urma_io.DATE_FORMAT

INPUT_DIR_ARG_NAME = 'input_urma_dir_name'
FIRST_DATE_ARG_NAME = 'first_valid_date_string'
LAST_DATE_ARG_NAME = 'last_valid_date_string'
NUM_DATES_ARG_NAME = 'num_valid_dates'
NUM_QUANTILES_ARG_NAME = 'num_quantiles'
NUM_SAMPLE_VALUES_ARG_NAME = 'num_sample_values_per_file'
OUTPUT_FILE_ARG_NAME = 'output_norm_file_name'

INPUT_DIR_HELP_STRING = (
    'Path to input directory.  Files therein will be found by '
    '`urma_io.find_file` and read by `urma_io.read_file`.'
)

FIRST_DATE_HELP_STRING = (
    'First valid date (format "yyyymmdd").  Normalization params will be based '
    'on all dates in the continuous period {0:s}...{1:s}.'
).format(FIRST_DATE_ARG_NAME, LAST_DATE_ARG_NAME)

LAST_DATE_HELP_STRING = 'See documentation for {0:s}.'.format(
    FIRST_DATE_ARG_NAME
)

NUM_DATES_HELP_STRING = (
    'Number of dates to use in computing z-score params.  These dates will be '
    'randomly sampled from the period {0:s}...{1:s}.'
).format(FIRST_DATE_ARG_NAME, LAST_DATE_ARG_NAME)

NUM_QUANTILES_HELP_STRING = (
    'Number of quantiles to store for each variable.  The quantile levels will '
    'be evenly spaced from 0 to 1 (i.e., the 0th to 100th percentile).'
)
NUM_SAMPLE_VALUES_HELP_STRING = (
    'Number of sample values per file to use for computing quantiles.  This '
    'value will be applied to each variable.'
)
OUTPUT_FILE_HELP_STRING = (
    'Path to output file.  Will be written by '
    '`urma_io.write_normalization_file`.'
)

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + INPUT_DIR_ARG_NAME, type=str, required=True,
    help=INPUT_DIR_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + FIRST_DATE_ARG_NAME, type=str, required=True,
    help=FIRST_DATE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + LAST_DATE_ARG_NAME, type=str, required=True,
    help=LAST_DATE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + NUM_DATES_ARG_NAME, type=int, required=True,
    help=NUM_DATES_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + NUM_QUANTILES_ARG_NAME, type=int, required=False, default=1001,
    help=NUM_QUANTILES_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + NUM_SAMPLE_VALUES_ARG_NAME, type=int, required=True,
    help=NUM_SAMPLE_VALUES_ARG_NAME
)
INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_FILE_ARG_NAME, type=str, required=True,
    help=OUTPUT_FILE_HELP_STRING
)


def _run(input_dir_name, first_valid_date_string, last_valid_date_string,
         num_valid_dates, num_quantiles, num_sample_values_per_file,
         output_file_name):
    """Computes normalization parameters for URMA target variables.

    This is effectively the main method.

    :param input_dir_name: See documentation at top of this script.
    :param first_valid_date_string: Same.
    :param last_valid_date_string: Same.
    :param num_valid_dates: Same.
    :param num_quantiles: Same.
    :param num_sample_values_per_file: Same.
    :param output_file_name: Same.
    """

    urma_file_names = urma_io.find_files_for_period(
        directory_name=input_dir_name,
        first_date_string=first_valid_date_string,
        last_date_string=last_valid_date_string,
        raise_error_if_any_missing=False,
        raise_error_if_all_missing=True
    )

    num_files = len(urma_file_names)
    file_indices = numpy.linspace(0, num_files - 1, num=num_files, dtype=int)

    if num_files > num_valid_dates:
        file_indices = numpy.random.choice(
            file_indices, size=num_valid_dates, replace=False
        )
        file_indices = numpy.sort(file_indices)
        urma_file_names = [urma_file_names[k] for k in file_indices]

    norm_param_table_xarray = (
        normalization.get_normalization_params_for_targets(
            urma_file_names=urma_file_names,
            num_quantiles=num_quantiles,
            num_sample_values_per_file=num_sample_values_per_file
        )
    )

    print('Writing z-score params to: "{0:s}"...'.format(output_file_name))
    urma_io.write_normalization_file(
        norm_param_table_xarray=norm_param_table_xarray,
        netcdf_file_name=output_file_name
    )


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        input_dir_name=getattr(INPUT_ARG_OBJECT, INPUT_DIR_ARG_NAME),
        first_valid_date_string=getattr(INPUT_ARG_OBJECT, FIRST_DATE_ARG_NAME),
        last_valid_date_string=getattr(INPUT_ARG_OBJECT, LAST_DATE_ARG_NAME),
        num_valid_dates=getattr(INPUT_ARG_OBJECT, NUM_DATES_ARG_NAME),
        num_quantiles=getattr(INPUT_ARG_OBJECT, NUM_QUANTILES_ARG_NAME),
        num_sample_values_per_file=getattr(
            INPUT_ARG_OBJECT, NUM_SAMPLE_VALUES_ARG_NAME
        ),
        output_file_name=getattr(INPUT_ARG_OBJECT, OUTPUT_FILE_ARG_NAME)
    )

"""Computes residual-normalization parameters for URMA target variables.

Residual-normalization parameters = stdev of temporal difference for each
variable.
"""

import os
import sys
import argparse
import numpy

THIS_DIRECTORY_NAME = os.path.dirname(os.path.realpath(
    os.path.join(os.getcwd(), os.path.expanduser(__file__))
))
sys.path.append(os.path.normpath(os.path.join(THIS_DIRECTORY_NAME, '..')))

import urma_io
import residual_normalization as resid_normalization

DATE_FORMAT = urma_io.DATE_FORMAT

INPUT_DIR_ARG_NAME = 'input_urma_dir_name'
NON_RESID_NORM_FILE_ARG_NAME = 'input_non_resid_norm_file_name'
FIRST_DATE_ARG_NAME = 'first_valid_date_string'
LAST_DATE_ARG_NAME = 'last_valid_date_string'
NUM_DATES_ARG_NAME = 'num_valid_dates'
OUTPUT_FILE_ARG_NAME = 'output_norm_file_name'

INPUT_DIR_HELP_STRING = (
    'Path to input directory.  Files therein will be found by '
    '`urma_io.find_file` and read by `urma_io.read_file`.'
)
NON_RESID_NORM_FILE_HELP_STRING = (
    'Path to file with parameters for non-residual normalization.'
)
FIRST_DATE_HELP_STRING = (
    'First valid date (format "yyyymmdd").  Normalization params will be based '
    'on all dates in the continuous period {0:s}...{1:s}.'
).format(
    FIRST_DATE_ARG_NAME, LAST_DATE_ARG_NAME
)
LAST_DATE_HELP_STRING = 'See documentation for {0:s}.'.format(
    FIRST_DATE_ARG_NAME
)
NUM_DATES_HELP_STRING = (
    'Number of dates to use in computing z-score params.  These dates will be '
    'randomly sampled from the period {0:s}...{1:s}.'
).format(
    FIRST_DATE_ARG_NAME, LAST_DATE_ARG_NAME
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
    '--' + NON_RESID_NORM_FILE_ARG_NAME, type=str, required=True,
    help=NON_RESID_NORM_FILE_HELP_STRING
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
    '--' + OUTPUT_FILE_ARG_NAME, type=str, required=True,
    help=OUTPUT_FILE_HELP_STRING
)


def _run(input_dir_name, non_resid_norm_file_name,
         first_valid_date_string, last_valid_date_string,
         num_valid_dates, output_file_name):
    """Computes normalization parameters for URMA target variables.

    This is effectively the main method.

    :param input_dir_name: See documentation at top of this script.
    :param non_resid_norm_file_name: Same.
    :param first_valid_date_string: Same.
    :param last_valid_date_string: Same.
    :param num_valid_dates: Same.
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
        resid_normalization.get_normalization_params_for_targets(
            urma_file_names=urma_file_names,
            non_resid_normalization_file_name=non_resid_norm_file_name
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
        non_resid_norm_file_name=getattr(
            INPUT_ARG_OBJECT, NON_RESID_NORM_FILE_ARG_NAME
        ),
        first_valid_date_string=getattr(INPUT_ARG_OBJECT, FIRST_DATE_ARG_NAME),
        last_valid_date_string=getattr(INPUT_ARG_OBJECT, LAST_DATE_ARG_NAME),
        num_valid_dates=getattr(INPUT_ARG_OBJECT, NUM_DATES_ARG_NAME),
        output_file_name=getattr(INPUT_ARG_OBJECT, OUTPUT_FILE_ARG_NAME)
    )

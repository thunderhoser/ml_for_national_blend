"""Runs discard test for each target field."""

import argparse
import numpy
from ml_for_national_blend.outside_code import time_conversion
from ml_for_national_blend.io import prediction_io
from ml_for_national_blend.utils import discard_test_utils as dt_utils

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'
TIME_FORMAT = '%Y-%m-%d-%H'

INPUT_DIR_ARG_NAME = 'input_prediction_dir_name'
INIT_TIME_LIMITS_ARG_NAME = 'init_time_limit_strings'
TARGET_FIELDS_ARG_NAME = 'target_field_names'
DISCARD_FRACTIONS_ARG_NAME = 'discard_fractions'
OUTPUT_FILE_ARG_NAME = 'output_file_name'

INPUT_DIR_HELP_STRING = (
    'Path to input directory, containing one prediction file per init time.  '
    'Files therein will be found by `prediction_io.find_file` and read by '
    '`prediction_io.read_file`.'
)
INIT_TIME_LIMITS_HELP_STRING = (
    'List of two initialization times, specifying the beginning and end of the '
    'evaluation period.  Time format is "yyyy-mm-dd-HH".'
)
TARGET_FIELDS_HELP_STRING = 'List of target fields to be evaluated.'
DISCARD_FRACTIONS_HELP_STRING = (
    'List of discard fractions, ranging from (0, 1).  This script will '
    'automatically use 0 as the lowest discard fraction.'
)
OUTPUT_FILE_HELP_STRING = (
    'Path to output file.  Evaluation scores will be written here by '
    '`spread_skill_utils.write_results`.'
)

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + INPUT_DIR_ARG_NAME, type=str, required=True,
    help=INPUT_DIR_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + INIT_TIME_LIMITS_ARG_NAME, type=str, nargs=2, required=True,
    help=INIT_TIME_LIMITS_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + TARGET_FIELDS_ARG_NAME, type=str, nargs='+', required=True,
    help=TARGET_FIELDS_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + DISCARD_FRACTIONS_ARG_NAME, type=float, nargs='+', required=True,
    help=DISCARD_FRACTIONS_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_FILE_ARG_NAME, type=str, required=True,
    help=OUTPUT_FILE_HELP_STRING
)


def _run(prediction_dir_name, init_time_limit_strings, target_field_names,
         discard_fractions, output_file_name):
    """Runs discard test for each target field.

    This is effectively the main method.

    :param prediction_dir_name: See documentation at top of this script.
    :param init_time_limit_strings: Same.
    :param target_field_names: Same.
    :param discard_fractions: Same.
    :param output_file_name: Same.
    """

    init_time_limits_unix_sec = numpy.array([
        time_conversion.string_to_unix_sec(t, TIME_FORMAT)
        for t in init_time_limit_strings
    ], dtype=int)

    prediction_file_names = prediction_io.find_files_for_period(
        directory_name=prediction_dir_name,
        first_init_time_unix_sec=init_time_limits_unix_sec[0],
        last_init_time_unix_sec=init_time_limits_unix_sec[1],
        raise_error_if_any_missing=False,
        raise_error_if_all_missing=True
    )

    result_table_xarray = dt_utils.run_discard_test(
        prediction_file_names=prediction_file_names,
        target_field_names=target_field_names,
        discard_fractions=discard_fractions,
        error_function=dt_utils.get_rmse_error_func_1field(),
        error_function_string='dt_utils.get_rmse_error_func_1field()',
        uncertainty_function=dt_utils.get_stdev_uncertainty_func_1field(),
        uncertainty_function_string=
        'dt_utils.get_stdev_uncertainty_func_1field()',
        is_error_pos_oriented=False
    )
    print(SEPARATOR_STRING)

    rtx = result_table_xarray
    target_field_names = rtx.coords[dt_utils.FIELD_DIM].values.tolist()

    for j in range(len(target_field_names)):
        print('Variable = {0:s} ... MF = {1:f} ... DI = {2:f}'.format(
            target_field_names[j],
            rtx[dt_utils.MONO_FRACTION_KEY].values[j],
            rtx[dt_utils.DISCARD_IMPROVEMENT_KEY].values[j]
        ))

    print('Writing results to: "{0:s}"...'.format(output_file_name))
    dt_utils.write_results(
        result_table_xarray=result_table_xarray,
        netcdf_file_name=output_file_name
    )


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        prediction_dir_name=getattr(INPUT_ARG_OBJECT, INPUT_DIR_ARG_NAME),
        init_time_limit_strings=getattr(
            INPUT_ARG_OBJECT, INIT_TIME_LIMITS_ARG_NAME
        ),
        target_field_names=getattr(
            INPUT_ARG_OBJECT, TARGET_FIELDS_ARG_NAME
        ),
        discard_fractions=numpy.array(
            getattr(INPUT_ARG_OBJECT, DISCARD_FRACTIONS_ARG_NAME), dtype=float
        ),
        output_file_name=getattr(INPUT_ARG_OBJECT, OUTPUT_FILE_ARG_NAME)
    )

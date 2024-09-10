"""Adds model name to NetCDF file with interpolated NWP data."""

import argparse
import numpy
import netCDF4
from ml_for_national_blend.outside_code import time_conversion
from ml_for_national_blend.outside_code import error_checking
from ml_for_national_blend.io import interp_nwp_model_io
from ml_for_national_blend.utils import nwp_model_utils

TIME_FORMAT = '%Y-%m-%d-%H'

INPUT_DIR_ARG_NAME = 'input_nwp_directory_name'
MODEL_ARG_NAME = 'nwp_model_name'
LEAD_TIMES_ARG_NAME = 'lead_times_hours'
FIRST_INIT_TIME_ARG_NAME = 'first_init_time_string'
LAST_INIT_TIME_ARG_NAME = 'last_init_time_string'

INPUT_DIR_HELP_STRING = (
    'Path to input directory.  Files therein will be found by '
    '`interp_nwp_model_io.find_file` and read by '
    '`interp_nwp_model_io.read_file`.'
)
MODEL_HELP_STRING = (
    'Name of NWP model.  Must be accepted by '
    '`nwp_model_utils.check_model_name`.'
)
LEAD_TIMES_HELP_STRING = (
    'List of lead times.  This script will process forecasts at every lead '
    'time in the list.'
)
FIRST_INIT_TIME_HELP_STRING = (
    'This script will process model runs initialized at all times in the '
    'period `{0:s}`...`{1:s}`.  Use the time format "yyyy-mm-dd-HH".'
).format(
    FIRST_INIT_TIME_ARG_NAME, LAST_INIT_TIME_ARG_NAME
)
LAST_INIT_TIME_HELP_STRING = FIRST_INIT_TIME_HELP_STRING

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + INPUT_DIR_ARG_NAME, type=str, required=True,
    help=INPUT_DIR_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + MODEL_ARG_NAME, type=str, required=True, help=MODEL_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + LEAD_TIMES_ARG_NAME, type=int, nargs='+', required=True,
    help=LEAD_TIMES_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + FIRST_INIT_TIME_ARG_NAME, type=str, required=True,
    help=FIRST_INIT_TIME_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + LAST_INIT_TIME_ARG_NAME, type=str, required=True,
    help=LAST_INIT_TIME_HELP_STRING
)


def _run(nwp_directory_name, model_name, lead_times_hours,
         first_init_time_string, last_init_time_string):
    """Adds model name to NetCDF file with interpolated NWP data.

    This is effectively the main method.

    :param nwp_directory_name: See documentation at top of this script.
    :param model_name: Same.
    :param lead_times_hours: Same.
    :param first_init_time_string: Same.
    :param last_init_time_string: Same.
    """

    # Check input args.
    if len(lead_times_hours) == 1 and lead_times_hours[0] < 0:
        if model_name == nwp_model_utils.RAP_MODEL_NAME:
            fake_init_time_unix_sec = 3 * 3600
        else:
            fake_init_time_unix_sec = 0

        lead_times_hours = nwp_model_utils.model_to_forecast_hours(
            model_name=model_name, init_time_unix_sec=fake_init_time_unix_sec
        )

    error_checking.assert_is_greater_numpy_array(lead_times_hours, 0)

    # Do actual stuff.
    first_init_time_unix_sec = time_conversion.string_to_unix_sec(
        first_init_time_string, TIME_FORMAT
    )
    last_init_time_unix_sec = time_conversion.string_to_unix_sec(
        last_init_time_string, TIME_FORMAT
    )

    for this_lead_time_hours in lead_times_hours:
        interp_nwp_file_names = interp_nwp_model_io.find_files_for_period(
            directory_name=nwp_directory_name,
            model_name=model_name,
            forecast_hour=this_lead_time_hours,
            first_init_time_unix_sec=first_init_time_unix_sec,
            last_init_time_unix_sec=last_init_time_unix_sec,
            raise_error_if_any_missing=False,
            raise_error_if_all_missing=False
        )

        for this_file_name in interp_nwp_file_names:
            print('Adding model name to: "{0:s}"...'.format(this_file_name))
            with netCDF4.Dataset(this_file_name, 'a') as this_dataset:
                this_dataset.setncattr('model_name', model_name)


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        nwp_directory_name=getattr(INPUT_ARG_OBJECT, INPUT_DIR_ARG_NAME),
        model_name=getattr(INPUT_ARG_OBJECT, MODEL_ARG_NAME),
        lead_times_hours=numpy.array(
            getattr(INPUT_ARG_OBJECT, LEAD_TIMES_ARG_NAME), dtype=int
        ),
        first_init_time_string=getattr(
            INPUT_ARG_OBJECT, FIRST_INIT_TIME_ARG_NAME
        ),
        last_init_time_string=getattr(
            INPUT_ARG_OBJECT, LAST_INIT_TIME_ARG_NAME
        )
    )

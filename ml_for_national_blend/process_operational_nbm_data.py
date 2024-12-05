"""Processes operational NBM data.

The operational NBM is used as a baseline to compare against our ML models.

Each raw file should be a GRIB2 file downloaded from Amazon Web Services (AWS)
with the following options:

- One model run (init time)
- One forecast hour (valid time)
- Full domain
- Full resolution
- Variables: all keys in list `raw_operational_nbm_io.ALL_FIELD_NAMES`

The output will contain the same data, in NetCDF format, with one file per init
time per valid time.
"""

import os
import sys
import argparse

THIS_DIRECTORY_NAME = os.path.dirname(os.path.realpath(
    os.path.join(os.getcwd(), os.path.expanduser(__file__))
))
sys.path.append(os.path.normpath(os.path.join(THIS_DIRECTORY_NAME, '..')))

import time_conversion
import time_periods
import raw_operational_nbm_io
import operational_nbm_io

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

HOURS_TO_SECONDS = 3600
TIME_FORMAT = '%Y-%m-%d-%H'

INPUT_DIR_ARG_NAME = 'input_grib2_dir_name'
FIRST_INIT_TIME_ARG_NAME = 'first_init_time_string'
LAST_INIT_TIME_ARG_NAME = 'last_init_time_string'
FORECAST_HOUR_ARG_NAME = 'forecast_hour'
WGRIB2_EXE_ARG_NAME = 'wgrib2_exe_file_name'
TEMPORARY_DIR_ARG_NAME = 'temporary_dir_name'
OUTPUT_DIR_ARG_NAME = 'output_netcdf_dir_name'

INPUT_DIR_HELP_STRING = (
    'Path to input directory, containing raw GRIB2 files.  Files therein will '
    'be found by `raw_operational_nbm_io.find_file`.'
)
FIRST_INIT_TIME_HELP_STRING = (
    'First init time (format "yyyy-mm-dd-HH").  This script will process model '
    'runs initialized at all times in the continuous period {0:s}...{1:s}.'
).format(
    FIRST_INIT_TIME_ARG_NAME, LAST_INIT_TIME_ARG_NAME
)
LAST_INIT_TIME_HELP_STRING = 'See documentation for {0:s}.'.format(
    FIRST_INIT_TIME_ARG_NAME
)
FORECAST_HOUR_HELP_STRING = (
    'Will process only files containing this forecast hour (lead time).'
)
WGRIB2_EXE_HELP_STRING = 'Path to wgrib2 executable.'
TEMPORARY_DIR_HELP_STRING = (
    'Path to temporary directory for text files created by wgrib2.'
)
OUTPUT_DIR_HELP_STRING = (
    'Path to output directory.  Processed files will be written here (one '
    'NetCDF file per init time per valid time) by '
    '`interp_nwp_model_io.write_file`, to exact locations determined by '
    '`interp_nwp_model_io.find_file`.'
)

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + INPUT_DIR_ARG_NAME, type=str, required=True,
    help=INPUT_DIR_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + FIRST_INIT_TIME_ARG_NAME, type=str, required=True,
    help=FIRST_INIT_TIME_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + LAST_INIT_TIME_ARG_NAME, type=str, required=True,
    help=LAST_INIT_TIME_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + FORECAST_HOUR_ARG_NAME, type=int, required=True,
    help=FORECAST_HOUR_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + WGRIB2_EXE_ARG_NAME, type=str, required=True,
    help=WGRIB2_EXE_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + TEMPORARY_DIR_ARG_NAME, type=str, required=True,
    help=TEMPORARY_DIR_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_DIR_ARG_NAME, type=str, required=True,
    help=OUTPUT_DIR_HELP_STRING
)


def _run(input_dir_name, first_init_time_string, last_init_time_string,
         forecast_hour, wgrib2_exe_name, temporary_dir_name,
         output_dir_name):
    """Processes operational NBM data.

    This is effectively the main method.

    :param input_dir_name: See documentation at top of this script.
    :param first_init_time_string: Same.
    :param last_init_time_string: Same.
    :param forecast_hour: Same.
    :param wgrib2_exe_name: Same.
    :param temporary_dir_name: Same.
    :param output_dir_name: Same.
    """

    first_init_time_unix_sec = time_conversion.string_to_unix_sec(
        first_init_time_string, TIME_FORMAT
    )
    last_init_time_unix_sec = time_conversion.string_to_unix_sec(
        last_init_time_string, TIME_FORMAT
    )
    init_times_unix_sec = time_periods.range_and_interval_to_list(
        start_time_unix_sec=first_init_time_unix_sec,
        end_time_unix_sec=last_init_time_unix_sec,
        time_interval_sec=HOURS_TO_SECONDS,
        include_endpoint=True
    )

    for this_init_time_unix_sec in init_times_unix_sec:
        input_file_name = raw_operational_nbm_io.find_file(
            directory_name=input_dir_name,
            init_time_unix_sec=this_init_time_unix_sec,
            forecast_hour=forecast_hour,
            raise_error_if_missing=False
        )
        print(input_file_name)

        if not os.path.isfile(input_file_name):
            continue

        op_nbm_forecast_table_xarray = raw_operational_nbm_io.read_file(
            grib2_file_name=input_file_name,
            wgrib2_exe_name=wgrib2_exe_name,
            temporary_dir_name=temporary_dir_name
        )
        print(SEPARATOR_STRING)

        output_file_name = operational_nbm_io.find_file(
            directory_name=output_dir_name,
            init_time_unix_sec=this_init_time_unix_sec,
            forecast_hour=forecast_hour,
            raise_error_if_missing=False
        )

        print('Writing data to: "{0:s}"...'.format(output_file_name))
        operational_nbm_io.write_file(
            op_nbm_forecast_table_xarray=op_nbm_forecast_table_xarray,
            netcdf_file_name=output_file_name
        )
        print(SEPARATOR_STRING)


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        input_dir_name=getattr(INPUT_ARG_OBJECT, INPUT_DIR_ARG_NAME),
        first_init_time_string=getattr(
            INPUT_ARG_OBJECT, FIRST_INIT_TIME_ARG_NAME
        ),
        last_init_time_string=getattr(
            INPUT_ARG_OBJECT, LAST_INIT_TIME_ARG_NAME
        ),
        forecast_hour=getattr(INPUT_ARG_OBJECT, FORECAST_HOUR_ARG_NAME),
        wgrib2_exe_name=getattr(INPUT_ARG_OBJECT, WGRIB2_EXE_ARG_NAME),
        temporary_dir_name=getattr(INPUT_ARG_OBJECT, TEMPORARY_DIR_ARG_NAME),
        output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )

"""Processes URMA data.

Each raw file should be a GRIB2 file downloaded from the NOAA High-performance
Storage System (HPSS) with the following options:

- One valid time
- Full domain
- Full resolution
- Variables: all keys in list `urma_utils.ALL_FIELD_NAMES`

The output will contain the same data, in NetCDF format, with one file per day.
"""

import os
import sys
import argparse
import numpy

THIS_DIRECTORY_NAME = os.path.dirname(os.path.realpath(
    os.path.join(os.getcwd(), os.path.expanduser(__file__))
))
sys.path.append(os.path.normpath(os.path.join(THIS_DIRECTORY_NAME, '..')))

import time_conversion
import time_periods
import urma_io
import raw_urma_io
import urma_utils

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'

HOURS_TO_SECONDS = 3600
DATE_FORMAT = '%Y%m%d'

INPUT_DIR_ARG_NAME = 'input_grib2_dir_name'
FIRST_DATE_ARG_NAME = 'first_valid_date_string'
LAST_DATE_ARG_NAME = 'last_valid_date_string'
WGRIB2_EXE_ARG_NAME = 'wgrib2_exe_file_name'
TEMPORARY_DIR_ARG_NAME = 'temporary_dir_name'
OUTPUT_DIR_ARG_NAME = 'output_netcdf_dir_name'

INPUT_DIR_HELP_STRING = (
    'Name of main input directory, containing one GRIB2 file per valid time.  '
    'Files therein will be found by `raw_urma_io.find_file`.'
)
FIRST_DATE_HELP_STRING = (
    'First date (format "yyyymmdd") to process.  This script will process URMA '
    'data for all days in the continuous period {0:s}...{1:s}.'
).format(
    FIRST_DATE_ARG_NAME, LAST_DATE_ARG_NAME
)
LAST_DATE_HELP_STRING = 'See documentation for {0:s}.'.format(
    FIRST_DATE_ARG_NAME
)
WGRIB2_EXE_HELP_STRING = 'Path to wgrib2 executable.'
TEMPORARY_DIR_HELP_STRING = (
    'Path to temporary directory for text files created by wgrib2.'
)
OUTPUT_DIR_HELP_STRING = (
    'Path to output directory.  Processed files will be written here (one '
    'NetCDF file per day) by `urma_io.write_file`, to exact locations '
    'determined by `urma_io.find_file`.'
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


def _run(input_dir_name, first_valid_date_string, last_valid_date_string,
         wgrib2_exe_name, temporary_dir_name, output_dir_name):
    """Processes NWP data.

    This is effectively the main method.

    :param input_dir_name: See documentation at top of this script.
    :param first_valid_date_string: Same.
    :param last_valid_date_string: Same.
    :param wgrib2_exe_name: Same.
    :param temporary_dir_name: Same.
    :param output_dir_name: Same.
    """

    valid_date_strings = time_conversion.get_spc_dates_in_range(
        first_valid_date_string, last_valid_date_string
    )

    latitude_matrix_deg_n = urma_utils.read_grid_coords()[0]
    num_grid_rows = latitude_matrix_deg_n.shape[0]
    num_grid_columns = latitude_matrix_deg_n.shape[1]

    desired_row_indices = numpy.linspace(
        0, num_grid_rows - 1, num=num_grid_rows, dtype=int
    )
    desired_column_indices = numpy.linspace(
        0, num_grid_columns - 1, num=num_grid_columns, dtype=int
    )
    field_names = urma_utils.ALL_FIELD_NAMES

    for this_date_string in valid_date_strings:
        this_date_unix_sec = time_conversion.string_to_unix_sec(
            this_date_string, DATE_FORMAT
        )
        valid_times_unix_sec = time_periods.range_and_interval_to_list(
            start_time_unix_sec=this_date_unix_sec,
            end_time_unix_sec=this_date_unix_sec + 23 * HOURS_TO_SECONDS,
            time_interval_sec=HOURS_TO_SECONDS,
            include_endpoint=True
        )

        input_file_names = [
            raw_urma_io.find_file(
                directory_name=input_dir_name,
                valid_time_unix_sec=t,
                raise_error_if_missing=True
            )
            for t in valid_times_unix_sec
        ]

        urma_tables_xarray = [None] * len(input_file_names)

        for k in range(len(input_file_names)):
            urma_tables_xarray[k] = raw_urma_io.read_file(
                grib2_file_name=input_file_names[k],
                desired_row_indices=desired_row_indices,
                desired_column_indices=desired_column_indices,
                wgrib2_exe_name=wgrib2_exe_name,
                temporary_dir_name=temporary_dir_name,
                field_names=field_names
            )
            print(SEPARATOR_STRING)

        urma_table_xarray = urma_utils.concat_over_time(urma_tables_xarray)

        output_file_name = urma_io.find_file(
            directory_name=output_dir_name,
            valid_date_string=this_date_string,
            raise_error_if_missing=False
        )

        print('Writing data to: "{0:s}"...'.format(output_file_name))
        urma_io.write_file(
            netcdf_file_name=output_file_name,
            urma_table_xarray=urma_table_xarray
        )
        print(SEPARATOR_STRING)


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        input_dir_name=getattr(INPUT_ARG_OBJECT, INPUT_DIR_ARG_NAME),
        first_valid_date_string=getattr(
            INPUT_ARG_OBJECT, FIRST_DATE_ARG_NAME
        ),
        last_valid_date_string=getattr(
            INPUT_ARG_OBJECT, LAST_DATE_ARG_NAME
        ),
        wgrib2_exe_name=getattr(INPUT_ARG_OBJECT, WGRIB2_EXE_ARG_NAME),
        temporary_dir_name=getattr(INPUT_ARG_OBJECT, TEMPORARY_DIR_ARG_NAME),
        output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )

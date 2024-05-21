"""Processes gridded MOS data.

Each raw file should be a TDL Pack file downloaded from the NOAA
High-performance Storage System (HPSS) with the following options:

- One month of model runs at a given initialization hour
  (e.g., all 00Z runs in Feb 2023)
- All forecast hours (valid times)
- Full domain
- Full resolution
- Variables: all keys in dict `FIELD_NAME_TO_TDLPACK_NAME` (defined below)

The output will contain the same data, in zarr format, with one file for every
individual model run (init time).
"""

import os
import shutil
import argparse
import numpy
from gewittergefahr.gg_utils import time_conversion
from gewittergefahr.gg_utils import time_periods
from ml_for_national_blend.io import nwp_model_io
from ml_for_national_blend.io import raw_gridded_mos_io
from ml_for_national_blend.utils import misc_utils
from ml_for_national_blend.utils import nwp_model_utils

SEPARATOR_STRING = '\n\n' + '*' * 50 + '\n\n'
TIME_FORMAT = '%Y-%m-%d-%H'

INPUT_DIR_ARG_NAME = 'input_tdlpack_dir_name'
FIRST_INIT_TIME_ARG_NAME = 'first_init_time_string'
LAST_INIT_TIME_ARG_NAME = 'last_init_time_string'
PROCESS_00Z_ARG_NAME = 'process_00z_runs'
PROCESS_12Z_ARG_NAME = 'process_12z_runs'
TAR_OUTPUTS_ARG_NAME = 'tar_output_files'
OUTPUT_DIR_ARG_NAME = 'output_zarr_dir_name'

INPUT_DIR_HELP_STRING = (
    'Path to input directory, containing one TDL Pack file for every init '
    'month/hour (e.g., 00Z runs in Jan 2023, 12Z runs in Jan 2023, '
    '00Z runs in Feb 2023, 12Z runs in Feb 2023, etc.).  '
    'Relevant files will be found by `raw_gridded_mos_io.find_file`.'
)
FIRST_INIT_TIME_HELP_STRING = (
    'First init time (format "yyyy-mm-dd-HH").  This script will process model '
    'runs initialized in the continuous period {0:s}...{1:s}.'
).format(
    FIRST_INIT_TIME_ARG_NAME, LAST_INIT_TIME_ARG_NAME
)
LAST_INIT_TIME_HELP_STRING = 'See documentation for {0:s}.'.format(
    FIRST_INIT_TIME_ARG_NAME
)
PROCESS_00Z_HELP_STRING = (
    'Boolean flag.  If 1, this script will process 00Z model runs in the '
    'period {0:s}...{1:s}.'
).format(
    FIRST_INIT_TIME_ARG_NAME, LAST_INIT_TIME_ARG_NAME
)
PROCESS_12Z_HELP_STRING = (
    'Boolean flag.  If 1, this script will process 12Z model runs in the '
    'period {0:s}...{1:s}.'
).format(
    FIRST_INIT_TIME_ARG_NAME, LAST_INIT_TIME_ARG_NAME
)
TAR_OUTPUTS_HELP_STRING = 'Boolean flag.  If 1, will tar output files.'
OUTPUT_DIR_HELP_STRING = (
    'Path to output directory.  Processed files will be written here (one '
    'zarr file per individual model run) by `nwp_model_io.write_file`, to '
    'exact locations determined by `nwp_model_io.find_file`.'
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
    '--' + PROCESS_00Z_ARG_NAME, type=int, required=True,
    help=PROCESS_00Z_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + PROCESS_12Z_ARG_NAME, type=int, required=True,
    help=PROCESS_12Z_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + TAR_OUTPUTS_ARG_NAME, type=int, required=False, default=0,
    help=TAR_OUTPUTS_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_DIR_ARG_NAME, type=str, required=True,
    help=OUTPUT_DIR_HELP_STRING
)


def _run(input_dir_name, first_init_time_string, last_init_time_string,
         process_00z_runs, process_12z_runs, tar_output_files, output_dir_name):
    """Processes gridded MOS data.

    This is effectively the main method.

    :param input_dir_name: See documentation at top of this script.
    :param first_init_time_string: Same.
    :param last_init_time_string: Same.
    :param process_00z_runs: Same.
    :param process_12z_runs: Same.
    :param tar_output_files: Same.
    :param output_dir_name: Same.
    """

    # Determine which time steps to read.
    assert process_00z_runs or process_12z_runs

    first_init_time_unix_sec = time_conversion.string_to_unix_sec(
        first_init_time_string, TIME_FORMAT
    )
    last_init_time_unix_sec = time_conversion.string_to_unix_sec(
        last_init_time_string, TIME_FORMAT
    )
    init_time_interval_sec = nwp_model_utils.model_to_init_time_interval(
        nwp_model_utils.GRIDDED_MOS_MODEL_NAME
    )

    init_times_unix_sec = time_periods.range_and_interval_to_list(
        start_time_unix_sec=first_init_time_unix_sec,
        end_time_unix_sec=last_init_time_unix_sec,
        time_interval_sec=init_time_interval_sec,
        include_endpoint=True
    )
    init_time_strings = [
        time_conversion.unix_sec_to_string(t, TIME_FORMAT)
        for t in init_times_unix_sec
    ]
    init_hours = [int(t.split('-')[-1]) for t in init_time_strings]

    if process_00z_runs and process_12z_runs:
        good_indices = numpy.linspace(
            0, len(init_hours) - 1, num=len(init_hours), dtype=int
        )
    elif process_00z_runs:
        good_indices = numpy.where(init_hours == 0)[0]
    else:
        good_indices = numpy.where(init_hours == 12)[0]

    init_times_unix_sec = init_times_unix_sec[good_indices]
    del init_time_strings
    del init_hours

    # Do actual stuff.
    for this_init_time_unix_sec in init_times_unix_sec:
        input_file_name = raw_gridded_mos_io.find_file(
            directory_name=input_dir_name,
            first_init_time_unix_sec=this_init_time_unix_sec,
            raise_error_if_missing=True
        )
        forecast_table_xarray = raw_gridded_mos_io.read_file(
            tdlpack_file_name=input_file_name,
            init_time_unix_sec=this_init_time_unix_sec
        )
        forecast_table_xarray = nwp_model_utils.remove_negative_precip(
            forecast_table_xarray
        )

        output_file_name = nwp_model_io.find_file(
            directory_name=output_dir_name,
            model_name=nwp_model_utils.GRIDDED_MOS_MODEL_NAME,
            init_time_unix_sec=this_init_time_unix_sec,
            raise_error_if_missing=False
        )

        print('Writing data to: "{0:s}"...'.format(output_file_name))
        nwp_model_io.write_file(
            zarr_file_name=output_file_name,
            nwp_forecast_table_xarray=forecast_table_xarray
        )
        print(SEPARATOR_STRING)

        if not tar_output_files:
            continue

        output_file_name_tarred = '{0:s}.tar'.format(
            os.path.splitext(output_file_name)[0]
        )
        print('Creating tar file: "{0:s}"...'.format(output_file_name_tarred))

        misc_utils.create_tar_file(
            source_paths_to_tar=[output_file_name],
            tar_file_name=output_file_name_tarred
        )
        shutil.rmtree(output_file_name)


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
        process_00z_runs=bool(getattr(INPUT_ARG_OBJECT, PROCESS_00Z_ARG_NAME)),
        process_12z_runs=bool(getattr(INPUT_ARG_OBJECT, PROCESS_12Z_ARG_NAME)),
        tar_output_files=bool(getattr(INPUT_ARG_OBJECT, TAR_OUTPUTS_ARG_NAME)),
        output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )

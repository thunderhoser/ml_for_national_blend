"""Interpolates NWP data from native grid to NBM grid."""

import os
import sys
import shutil
import argparse

THIS_DIRECTORY_NAME = os.path.dirname(os.path.realpath(
    os.path.join(os.getcwd(), os.path.expanduser(__file__))
))
sys.path.append(os.path.normpath(os.path.join(THIS_DIRECTORY_NAME, '..')))

import time_conversion
import nwp_model_io
import misc_utils
import nwp_model_utils

TIME_FORMAT = '%Y-%m-%d-%H'

INPUT_DIR_ARG_NAME = 'input_native_grid_dir_name'
MODEL_ARG_NAME = 'model_name'
FIRST_INIT_TIME_ARG_NAME = 'first_init_time_string'
LAST_INIT_TIME_ARG_NAME = 'last_init_time_string'
TAR_OUTPUTS_ARG_NAME = 'tar_output_files'
OUTPUT_DIR_ARG_NAME = 'output_nbm_grid_dir_name'

INPUT_DIR_HELP_STRING = (
    'Path to input directory.  Data on native model grid will be found in this '
    'directory by `nwp_model_io.find_file`.'
)
MODEL_HELP_STRING = (
    'Name of NWP model (must be accepted by '
    '`nwp_model_utils.check_model_name`).'
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
TAR_OUTPUTS_HELP_STRING = 'Boolean flag.  If 1, will tar output files.'
OUTPUT_DIR_HELP_STRING = (
    'Path to output directory.  Data on NBM grid will be written here (one '
    'NetCDF file per model run per lead time) by `nwp_model_io.write_file`, '
    'to exact locations determined by `nwp_model_io.find_file`.'
)

INPUT_ARG_PARSER = argparse.ArgumentParser()
INPUT_ARG_PARSER.add_argument(
    '--' + INPUT_DIR_ARG_NAME, type=str, required=True,
    help=INPUT_DIR_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + MODEL_ARG_NAME, type=str, required=True, help=MODEL_HELP_STRING
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
    '--' + TAR_OUTPUTS_ARG_NAME, type=int, required=False, default=0,
    help=TAR_OUTPUTS_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_DIR_ARG_NAME, type=str, required=True,
    help=OUTPUT_DIR_HELP_STRING
)


def _run(input_dir_name, model_name, first_init_time_string,
         last_init_time_string, tar_output_files, output_dir_name):
    """Interpolates NWP data from native grid to NBM grid.

    This is effectively the main method.

    :param input_dir_name: See documentation at top of this script.
    :param model_name: Same.
    :param first_init_time_string: Same.
    :param last_init_time_string: Same.
    :param tar_output_files: Same.
    :param output_dir_name: Same.
    """

    first_init_time_unix_sec = time_conversion.string_to_unix_sec(
        first_init_time_string, TIME_FORMAT
    )
    last_init_time_unix_sec = time_conversion.string_to_unix_sec(
        last_init_time_string, TIME_FORMAT
    )
    input_file_names = nwp_model_io.find_files_for_period(
        directory_name=input_dir_name,
        model_name=model_name,
        first_init_time_unix_sec=first_init_time_unix_sec,
        last_init_time_unix_sec=last_init_time_unix_sec,
        allow_tar=True,
        raise_error_if_any_missing=False,
        raise_error_if_all_missing=True
    )

    for this_input_file_name in input_file_names:
        print('Reading data on native grid from: "{0:s}"...'.format(
            this_input_file_name
        ))
        nwp_forecast_table_xarray = nwp_model_io.read_file(
            zarr_file_name=this_input_file_name,
            allow_tar=True
        )

        nwp_forecast_table_xarray = nwp_model_utils.interp_data_to_nbm_grid(
            nwp_forecast_table_xarray=nwp_forecast_table_xarray,
            model_name=model_name,
            use_nearest_neigh=True
        )

        output_file_name = nwp_model_io.find_file(
            directory_name=output_dir_name,
            model_name=model_name,
            init_time_unix_sec=
            nwp_model_io.file_name_to_init_time(this_input_file_name),
            raise_error_if_missing=False
        )

        print('Writing interpolated data to: "{0:s}"...'.format(
            output_file_name
        ))
        nwp_model_io.write_file(
            nwp_forecast_table_xarray=nwp_forecast_table_xarray,
            zarr_file_name=output_file_name
        )

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
        model_name=getattr(INPUT_ARG_OBJECT, MODEL_ARG_NAME),
        first_init_time_string=getattr(
            INPUT_ARG_OBJECT, FIRST_INIT_TIME_ARG_NAME
        ),
        last_init_time_string=getattr(
            INPUT_ARG_OBJECT, LAST_INIT_TIME_ARG_NAME
        ),
        tar_output_files=bool(getattr(INPUT_ARG_OBJECT, TAR_OUTPUTS_ARG_NAME)),
        output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )

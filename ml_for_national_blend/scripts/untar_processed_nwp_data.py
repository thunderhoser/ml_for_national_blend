"""Untars processed NWP data."""

import os
import argparse
from gewittergefahr.gg_utils import time_conversion
from gewittergefahr.gg_utils import time_periods
from ml_for_national_blend.io import nwp_model_io
from ml_for_national_blend.utils import nwp_model_utils
from ml_for_national_blend.utils import misc_utils

TIME_FORMAT = '%Y-%m-%d-%H'

INPUT_DIR_ARG_NAME = 'input_tar_dir_name'
MODEL_ARG_NAME = 'model_name'
FIRST_INIT_TIME_ARG_NAME = 'first_init_time_string'
LAST_INIT_TIME_ARG_NAME = 'last_init_time_string'
DELETE_ORIG_FILES_ARG_NAME = 'delete_original_files'
OUTPUT_DIR_ARG_NAME = 'output_zarr_dir_name'

INPUT_DIR_HELP_STRING = (
    'Path to input directory, containing one tar file per model run (init '
    'time).  Files therein will be found by `nwp_model_io.find_file`, but with '
    'the extension ".zarr" replaced by ".tar".'
)
MODEL_HELP_STRING = (
    'Name of NWP model (must be accepted by '
    '`nwp_model_utils.check_model_name`).'
)
FIRST_INIT_TIME_HELP_STRING = (
    'First init time (format "yyyy-mm-dd-HH").  This script will untar model '
    'runs for all times in the continuous period {0:s}...{1:s}.'
).format(
    FIRST_INIT_TIME_ARG_NAME, LAST_INIT_TIME_ARG_NAME
)
LAST_INIT_TIME_HELP_STRING = 'See documentation for {0:s}.'.format(
    FIRST_INIT_TIME_ARG_NAME
)
DELETE_ORIG_FILES_HELP_STRING = (
    'Boolean flag.  If 1, will delete original tar files after untarring.'
)
OUTPUT_DIR_HELP_STRING = (
    'Path to output directory.  Zarr files will be written here (one file per '
    'model run).'
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
    '--' + DELETE_ORIG_FILES_ARG_NAME, type=int, required=True,
    help=DELETE_ORIG_FILES_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_DIR_ARG_NAME, type=str, required=True,
    help=OUTPUT_DIR_HELP_STRING
)


def _run(input_dir_name, model_name, first_init_time_string,
         last_init_time_string, delete_original_files, output_dir_name):
    """Untars processed NWP data.

    This is effectively the main method.

    :param input_dir_name: See documentation at top of this script.
    :param model_name: Same.
    :param first_init_time_string: Same.
    :param last_init_time_string: Same.
    :param delete_original_files: Same.
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
        time_interval_sec=nwp_model_utils.model_to_init_time_interval(
            model_name
        ),
        include_endpoint=True
    )

    for this_init_time_unix_sec in init_times_unix_sec:
        zarr_file_name = nwp_model_io.find_file(
            directory_name=input_dir_name,
            model_name=model_name,
            init_time_unix_sec=this_init_time_unix_sec,
            raise_error_if_missing=False
        )
        tar_file_name = '{0:s}/{1:s}.tar'.format(
            input_dir_name,
            os.path.splitext(os.path.split(zarr_file_name)[1])[0]
        )

        print('Untarring file to (hopefully): "{0:s}"...'.format(
            zarr_file_name
        ))
        misc_utils.untar_zarr_or_netcdf_file(
            tar_file_name=tar_file_name,
            target_dir_name=output_dir_name
        )

        if delete_original_files:
            os.remove(tar_file_name)


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
        delete_original_files=bool(getattr(
            INPUT_ARG_OBJECT, DELETE_ORIG_FILES_ARG_NAME
        )),
        output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )

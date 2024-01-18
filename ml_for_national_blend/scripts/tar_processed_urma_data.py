"""Tars processed URMA data."""

import os
import shutil
import argparse
from gewittergefahr.gg_utils import time_conversion
from gewittergefahr.gg_utils import file_system_utils
from ml_for_national_blend.io import urma_io
from ml_for_national_blend.utils import misc_utils

TIME_FORMAT = '%Y-%m-%d-%H'

INPUT_DIR_ARG_NAME = 'input_zarr_dir_name'
FIRST_DATE_ARG_NAME = 'first_valid_date_string'
LAST_DATE_ARG_NAME = 'last_valid_date_string'
DELETE_ORIG_FILES_ARG_NAME = 'delete_original_files'
OUTPUT_DIR_ARG_NAME = 'output_tar_dir_name'

INPUT_DIR_HELP_STRING = (
    'Path to input directory, containing one zarr file per valid date.  Files '
    'therein will be found by `urma_io.find_file`.'
)
FIRST_DATE_HELP_STRING = (
    'First date (format "yyyymmdd") to tar.  This script will tar URMA data '
    'for all days in the continuous period {0:s}...{1:s}.'
).format(
    FIRST_DATE_ARG_NAME, LAST_DATE_ARG_NAME
)
LAST_DATE_HELP_STRING = 'See documentation for {0:s}.'.format(
    FIRST_DATE_ARG_NAME
)
DELETE_ORIG_FILES_HELP_STRING = (
    'Boolean flag.  If 1, will delete original zarr files after tarring.'
)
OUTPUT_DIR_HELP_STRING = (
    'Path to output directory.  Tar files will be written here (one file per '
    'valid date).'
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
    '--' + DELETE_ORIG_FILES_ARG_NAME, type=int, required=True,
    help=DELETE_ORIG_FILES_HELP_STRING
)
INPUT_ARG_PARSER.add_argument(
    '--' + OUTPUT_DIR_ARG_NAME, type=str, required=True,
    help=OUTPUT_DIR_HELP_STRING
)


def _run(input_dir_name, first_valid_date_string, last_valid_date_string,
         delete_original_files, output_dir_name):
    """Tars processed URMA data.

    This is effectively the main method.

    :param input_dir_name: See documentation at top of this script.
    :param first_valid_date_string: Same.
    :param last_valid_date_string: Same.
    :param delete_original_files: Same.
    :param output_dir_name: Same.
    """

    valid_date_strings = time_conversion.get_spc_dates_in_range(
        first_valid_date_string, last_valid_date_string
    )

    file_system_utils.mkdir_recursive_if_necessary(
        directory_name=output_dir_name
    )

    for this_date_string in valid_date_strings:
        zarr_file_name = urma_io.find_file(
            directory_name=input_dir_name,
            valid_date_string=this_date_string,
            raise_error_if_missing=True
        )
        tar_file_name = '{0:s}/{1:s}.tar'.format(
            output_dir_name,
            os.path.splitext(os.path.split(zarr_file_name)[1])[0]
        )

        print('Creating tar file: "{0:s}"...'.format(tar_file_name))
        misc_utils.create_tar_file(
            source_paths_to_tar=[zarr_file_name],
            tar_file_name=tar_file_name
        )

        if delete_original_files:
            shutil.rmtree(zarr_file_name)


if __name__ == '__main__':
    INPUT_ARG_OBJECT = INPUT_ARG_PARSER.parse_args()

    _run(
        input_dir_name=getattr(INPUT_ARG_OBJECT, INPUT_DIR_ARG_NAME),
        first_valid_date_string=getattr(INPUT_ARG_OBJECT, FIRST_DATE_ARG_NAME),
        last_valid_date_string=getattr(INPUT_ARG_OBJECT, LAST_DATE_ARG_NAME),
        delete_original_files=bool(getattr(
            INPUT_ARG_OBJECT, DELETE_ORIG_FILES_ARG_NAME
        )),
        output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )

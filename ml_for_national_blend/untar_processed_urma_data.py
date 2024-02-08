"""Untars processed URMA data."""

import os
import sys
import argparse

THIS_DIRECTORY_NAME = os.path.dirname(os.path.realpath(
    os.path.join(os.getcwd(), os.path.expanduser(__file__))
))
sys.path.append(os.path.normpath(os.path.join(THIS_DIRECTORY_NAME, '..')))

import time_conversion
import urma_io
import misc_utils

TIME_FORMAT = '%Y-%m-%d-%H'

INPUT_DIR_ARG_NAME = 'input_tar_dir_name'
FIRST_DATE_ARG_NAME = 'first_valid_date_string'
LAST_DATE_ARG_NAME = 'last_valid_date_string'
DELETE_ORIG_FILES_ARG_NAME = 'delete_original_files'
OUTPUT_DIR_ARG_NAME = 'output_zarr_dir_name'

INPUT_DIR_HELP_STRING = (
    'Path to input directory, containing one tar file per valid date.  Files '
    'therein will be found by `urma_io.find_file`, but with the extension '
    '".zarr" replaced by ".tar".'
)
FIRST_DATE_HELP_STRING = (
    'First date (format "yyyymmdd") to untar.  This script will untar URMA data '
    'for all days in the continuous period {0:s}...{1:s}.'
).format(
    FIRST_DATE_ARG_NAME, LAST_DATE_ARG_NAME
)
LAST_DATE_HELP_STRING = 'See documentation for {0:s}.'.format(
    FIRST_DATE_ARG_NAME
)
DELETE_ORIG_FILES_HELP_STRING = (
    'Boolean flag.  If 1, will delete original tar files after untarring.'
)
OUTPUT_DIR_HELP_STRING = (
    'Path to output directory.  Zarr files will be written here (one file per '
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
    """Untars processed URMA data.

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

    for this_date_string in valid_date_strings:
        zarr_file_name = urma_io.find_file(
            directory_name=input_dir_name,
            valid_date_string=this_date_string,
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
        first_valid_date_string=getattr(INPUT_ARG_OBJECT, FIRST_DATE_ARG_NAME),
        last_valid_date_string=getattr(INPUT_ARG_OBJECT, LAST_DATE_ARG_NAME),
        delete_original_files=bool(getattr(
            INPUT_ARG_OBJECT, DELETE_ORIG_FILES_ARG_NAME
        )),
        output_dir_name=getattr(INPUT_ARG_OBJECT, OUTPUT_DIR_ARG_NAME)
    )
